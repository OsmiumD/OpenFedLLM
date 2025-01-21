"""
Usage:
python merge_soft_prompt.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH] --soft_prompt [SOFT-PROMPT-ID]
"""
import argparse
import os.path
import torch

from peft import PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from soft_prompt_model import PeftModelAdditionPrompt


def merge_lora(base_model_name, lora_path, soft_prompt):

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    lora_model = PeftModel.from_pretrained(base_model, os.path.join(lora_path, 'lora'))
    client_model = PeftModel.from_pretrained(lora_model, os.path.join(lora_path, 'soft_prompt'))
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = client_model.merge_and_unload()
    soft_prompt_dict = torch.load(os.path.join(lora_path, 'soft-prompts', f'soft-prompt-{soft_prompt}.pt'))
    set_peft_model_state_dict(client_model, soft_prompt_dict)
    target_model_path = lora_path.replace("checkpoint", "full")
    target_model_path = target_model_path + f"-soft-prompt-{soft_prompt}"
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument('--soft_prompt', type=int, default=0)

    args = parser.parse_args()

    merge_lora(args.base_model_path, args.lora_path, args.soft_prompt)