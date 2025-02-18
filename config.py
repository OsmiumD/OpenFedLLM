from dataclasses import dataclass, field, asdict
from typing import Optional, List
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PromptEncoderConfig
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime, timedelta


# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=500, metadata={"help": "the number of rounds"})
    num_clients_dataset: Optional[int] = field(default=2, metadata={"help": "the number of clients per dataset"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    balance_sample: Optional[bool] = field(default=False, metadata={"help": "sample with average number of dataset"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    save_model_freq: Optional[int] = field(default=50, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_names: Optional[List[str]] = field(default_factory=list,metadata={"help": "Dataset names for each dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})    # vicuna and alpaca use 2e-5
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    stage_one_steps: Optional[int] = field(default=100, metadata={"help": "the number of stage_one steps"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    template_dataset: Optional[List[str]] = field(default_factory=list,metadata={"help": "Specified template for each dataset"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})
    dataset_sample: Optional[int] = field(default=20000, metadata={"help": "the number of samples to use from the dataset"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})
    use_client_model: Optional[bool] = field(default=False, metadata={"help": "use soft prompt for each client"})
    client_model_type: Optional[str] = field(default="soft-prompt", metadata={"help": "the model type for client"})
    soft_prompt_size: Optional[int] = field(default=20, metadata={"help": "size of soft prompt"})
    client_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    client_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    stage_one_lr: Optional[float] = field(default=1e-3, metadata={"help": "the learning rate in stage one"})
    stage_one_train: Optional[bool] = field(default=False, metadata={"help": "train stage one"})
    freeze_stage_one: Optional[bool] = field(default=False, metadata={"help": "whether to freeze stage one"})
    stage_one_ckpt: Optional[str] = field(default=None, metadata={"help": "whether to freeze stage one"})
    two_stage_training: Optional[bool] = field(default=False, metadata={"help": "Run two stage training"})

parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()
fed_args.num_datasets = len(script_args.dataset_names)
fed_args.num_clients = fed_args.num_clients_dataset * len(script_args.dataset_names)
script_args.dataset_name = ""
for ds_name in script_args.dataset_names:
    ds_name = os.path.basename(ds_name)
    script_args.dataset_name += ds_name[:3] + '-'
script_args.dataset_name = script_args.dataset_name[:-1]

# ===== Define the LoraConfig =====
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    client_peft_config = LoraConfig(
        r=script_args.client_lora_r,
        lora_alpha=script_args.client_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None
    client_peft_config = None


def get_config():
    return script_args, fed_args, peft_config

def get_client_peft_config():
    return client_peft_config


# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
    )
    return training_args

def get_stage_training_args(script_args):
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.stage_one_lr,
        logging_steps=10,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.stage_one_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
    )
    return training_args


def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    elif script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = {"": Accelerator().local_process_index}
        quantization_config = None
        torch_dtype = torch.bfloat16
    return device_map, quantization_config, torch_dtype


def save_config(script_args, fed_args):
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    dataset_name_split = script_args.dataset_name
    output_dir = f"{script_args.output_dir}/{dataset_name_split}_{script_args.dataset_sample}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
    while True:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            break
        else:
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_{now_time}"

    script_args.output_dir = output_dir
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
        }
        json.dump(combined_dict, f, indent=4)
