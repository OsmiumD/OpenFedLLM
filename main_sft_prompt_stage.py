import copy
import os
import sys

import torch
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, \
    PromptEncoderConfig, PromptTuningConfig, PeftConfig, PeftModel, LoraModel

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args, get_stage_training_args
from utils.freeze_adapter import unfreeze_adapter

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)
if not os.path.exists(os.path.join('./', script_args.output_dir)):
    os.makedirs(os.path.join('./', script_args.output_dir))
use_soft_prompt = script_args.use_client_model

# ===== Load the dataset =====
if len(script_args.dataset_names) == 0:
    print('no dataset specified, exiting.', file=sys.stderr)
    exit(0)
datasets = [process_sft_dataset(ds_name, get_dataset(ds_name, script_args.local_data_dir), script_args.dataset_sample)
            for ds_name in script_args.dataset_names]
# ===== Split the dataset into clients =====
local_datasets = [sp for ds in datasets for sp in split_dataset(fed_args, script_args, ds)]
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

base_model_copy = copy.deepcopy(base_model)

soft_prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",  # Task type: causal LM
    num_virtual_tokens=script_args.soft_prompt_size,  # Number of learnable tokens to prepend
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )

model_combined = get_addition_prompt_module(base_model, peft_config, 'global')
if use_soft_prompt:
    model_combined.add_soft_prompt_adapter('client_specific', soft_prompt_config)
model_combined.print_trainable_parameters()

model_soft_prompt = get_peft_model(base_model_copy, soft_prompt_config, 'client_specific')


model_combined.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model_soft_prompt.config.use_cache = False

if training_args.gradient_checkpointing:
    model_combined.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model_combined, adapter_name='global'))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]

local_soft_prompt = []
if use_soft_prompt:
    print('>> ==================== Using soft prompt for each client ==================== ')
    init_soft_prompt = copy.deepcopy(get_peft_model_state_dict(model_soft_prompt, adapter_name='client_specific'))
    local_soft_prompt = [copy.deepcopy(init_soft_prompt) for i in range(fed_args.num_clients)]

proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token  # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[
                        2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
formatting_prompts_func_spec = []
if len(script_args.template_dataset) == len(script_args.dataset_names):
    print(' ==================== Using specified template for each dataset ==================== ')
    for template_name in script_args.template_dataset:
        format_spec, _ = get_formatting_prompts_func(template_name, tokenizer.eos_token)
        formatting_prompts_func_spec.append(format_spec)
    assert len(formatting_prompts_func_spec) == len(script_args.dataset_names)

# ===== Training soft prompts for each client before global training =====
if use_soft_prompt:
    for name, param in model_soft_prompt.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    model_soft_prompt.print_trainable_parameters()
    for client in range(fed_args.num_clients):
        print('>> ==================== Client', client, '======================= ')
        set_peft_model_state_dict(model_soft_prompt, local_soft_prompt[client], 'client_specific')
        training_args = get_stage_training_args(script_args)

        # ===== Train local model on the client side =====
        trainer = SFTTrainer(
            model=model_soft_prompt,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_datasets[client],
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )

        results = trainer.train()
        local_soft_prompt[client] = copy.deepcopy(get_peft_model_state_dict(model_soft_prompt, adapter_name='client_specific'))

# ===== Start federated training =====
freeze_adapter(model_combined, 'client_specific')
unfreeze_adapter(model_combined, 'global')
model_combined.print_trainable_parameters()
training_loss = np.zeros((fed_args.num_clients, fed_args.num_rounds))
for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)
    training_loss[:, round] = -1
    print(f">> ==================== Round {round + 1} : {clients_this_round} ====================")

    for client in clients_this_round:
        set_peft_model_state_dict(model_combined, global_dict, 'global')  # sync the global model to the local model
        if use_soft_prompt:
            set_peft_model_state_dict(model_combined, local_soft_prompt[client], 'client_specific')  # add client local soft prompt

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args,
                                             script_args)  # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate,
                                      1e-6)  # manually schedule the learning rate
        print(f'learning rate: {new_lr}')
        training_args = get_training_args(script_args, new_lr)
        local_format_func = formatting_prompts_func
        if len(formatting_prompts_func_spec) == len(script_args.dataset_names):
            print(' ==================== Using specified template ==================== ')
            local_format_func = formatting_prompts_func_spec[client // fed_args.num_clients_dataset]

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model_combined,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=local_format_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        results = trainer.train()
        training_loss[client, round] = results.training_loss

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model_combined, adapter_name='global'))  # copy is needed!

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model_combined, global_dict, adapter_name='global')  # Update global model

    # ===== Save the model =====

    if (round + 1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}"))
        if use_soft_prompt:
            os.makedirs(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}", 'clients'))
            for idx, soft_prompt in enumerate(local_soft_prompt):
                torch.save(soft_prompt, os.path.join(script_args.output_dir, f"checkpoint-{round + 1}", 'clients', f'soft-prompt-{idx}.pt'))



    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), training_loss)
