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
from config import get_config, save_config, get_model_config, get_training_args, get_stage_training_args, \
    get_client_peft_config
from utils.freeze_adapter import unfreeze_adapter

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
print(script_args, fed_args)
if not os.path.exists(os.path.join('./', script_args.output_dir)):
    os.makedirs(os.path.join('./', script_args.output_dir))
use_client_model = script_args.use_client_model
torch.set_autocast_gpu_dtype(torch.bfloat16)

# ===== Load the dataset =====
if len(script_args.dataset_names) == 0:
    print('no dataset specified, exiting.', file=sys.stderr)
    exit(0)
datasets = [process_sft_dataset(ds_name, get_dataset(ds_name, script_args.local_data_dir, script_args.local_dataset),
                                script_args.dataset_sample * fed_args.num_clients_dataset)
            for ds_name in script_args.dataset_names]
datasets = [ds for ds_list in datasets for ds in (ds_list if isinstance(ds_list, list) else [ds_list])]
# ===== Split the dataset into clients =====
fed_args.num_datasets = len(datasets)
fed_args.num_clients = fed_args.num_clients_dataset * len(datasets)
local_datasets = [sp for ds in datasets for sp in split_dataset(fed_args, script_args, ds)]
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
save_config(script_args, fed_args)

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

if script_args.client_model_type == 'soft-prompt':
    client_side_config = PromptTuningConfig(
        task_type="CAUSAL_LM",  # Task type: causal LM
        num_virtual_tokens=script_args.soft_prompt_size,  # Number of learnable tokens to prepend
        prompt_tuning_init='TEXT',
        prompt_tuning_init_text='You are a helpful assistant',
        tokenizer_name_or_path=script_args.model_name_or_path
    )
elif script_args.client_model_type == 'lora':
    client_side_config = get_client_peft_config()
else:
    print('Unrecognized client model type.', file=sys.stderr)
    exit(0)

local_model_dict = []

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

# ===== Training client model for each client before global training =====
if use_client_model:
    if script_args.stage_one_ckpt:
        print('>> ==================== Loading client checkpoints ======================= ')
        for i in range(fed_args.num_clients):
            local_model_dict.append(
                torch.load(os.path.join(script_args.stage_one_ckpt, f'client-{i}.pt'), map_location='cuda'))
    elif script_args.two_stage_training:
        model_stage_one = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map=None,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=torch_dtype,
        )

        stage_one_loss = []

        if script_args.load_in_8bit or script_args.load_in_4bit:
            model_stage_one = prepare_model_for_kbit_training(
                model_stage_one, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model_stage_one = get_peft_model(model_stage_one, client_side_config, 'client_specific')
        model_stage_one.config.use_cache = False
        model_stage_one.enable_input_require_grads()

        init_soft_prompt = copy.deepcopy(get_peft_model_state_dict(model_stage_one, adapter_name='client_specific'))
        local_model_dict = [copy.deepcopy(init_soft_prompt) for i in range(fed_args.num_clients)]

        for name, param in model_stage_one.named_parameters():
            if param.requires_grad:
                print(name, param.size())
        model_stage_one = model_stage_one.to('cuda')
        os.makedirs(os.path.join(script_args.output_dir, f"before-fed"))
        model_stage_one.print_trainable_parameters()
        for client in range(fed_args.num_clients):
            print('>> ==================== Client', client, '======================= ')
            set_peft_model_state_dict(model_stage_one, local_model_dict[client], 'client_specific')
            training_args = get_stage_training_args(script_args)
            model_stage_one.print_trainable_parameters()
            # ===== Train local model on the client side =====
            trainer = SFTTrainer(
                model=model_stage_one,
                tokenizer=tokenizer,
                args=training_args,
                max_seq_length=script_args.seq_length,
                train_dataset=local_datasets[client],
                formatting_func=formatting_prompts_func,
                data_collator=data_collator,
            )

            results = trainer.train()
            local_model_dict[client] = copy.deepcopy(
                get_peft_model_state_dict(model_stage_one, adapter_name='client_specific'))

            client_loss = []
            for log in trainer.state.log_history:
                if 'loss' in log.keys():
                    client_loss.append(log['loss'])
            stage_one_loss.append(client_loss)
            np.save(os.path.join(script_args.output_dir, "stage_one_loss.npy"), np.array(stage_one_loss))

            torch.save(local_model_dict[client],
                       os.path.join(script_args.output_dir, f"before-fed", f'client-{client}.pt'))
            print('>> ==================== Client', client, ' model saved ======================= ')

model_combined = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=None,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model_combined = prepare_model_for_kbit_training(
        model_combined, use_gradient_checkpointing=training_args.gradient_checkpointing
    )

model_combined = get_addition_prompt_module(model_combined, peft_config, 'global')
print('model device: ', model_combined.device)
if use_client_model:
    if script_args.client_model_type == 'soft-prompt':
        model_combined.add_soft_prompt_adapter('client_specific', client_side_config)
    elif script_args.client_model_type == 'lora':
        model_combined.add_adapter('client_specific', client_side_config)
model_combined.print_trainable_parameters()
model_combined.enable_input_require_grads()

model_combined.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model_combined = model_combined.to('cuda')

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model_combined, adapter_name='global'))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]

proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

if len(local_model_dict) == 0 and use_client_model:  # Use client model but still not initialized
    init_soft_prompt = copy.deepcopy(get_peft_model_state_dict(model_combined, adapter_name='client_specific'))
    local_model_dict = [copy.deepcopy(init_soft_prompt) for i in range(fed_args.num_clients)]

# ===== Start federated training =====
if script_args.freeze_stage_one:
    print('>> ==================== Freezing stage one params ======================= ')
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
        if use_client_model:
            set_peft_model_state_dict(model_combined, local_model_dict[client],
                                      'client_specific')  # add client local soft prompt

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

        optimizers = (None, None)
        if script_args.separate_lr:
            optim = torch.optim.AdamW(
                group_parameters_by_layer(model_combined, new_lr, script_args.stage_one_lr, 'global',
                                          'client-specific'))
            # scheduler = get_scheduler(
            #     self.args.lr_scheduler_type,
            #     optimizer=self.optimizer if optimizer is None else optimizer,
            #     num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            #     num_training_steps=num_training_steps,
            # )
            optimizers = (optim, None)

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
            optimizers=optimizers
        )

        results = trainer.train()
        training_loss[client, round] = results.training_loss

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(
            get_peft_model_state_dict(model_combined, adapter_name='global'))  # copy is needed!

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
        if use_client_model:
            os.makedirs(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}", 'clients'))
            for idx, local_dict in enumerate(local_model_dict):
                torch.save(local_dict, os.path.join(script_args.output_dir, f"checkpoint-{round + 1}", 'clients',
                                                    f'client-{idx}.pt'))

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), training_loss)
