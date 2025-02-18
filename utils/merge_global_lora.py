"""
Usage:
python merge_global_lora.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH] --soft_prompt [SOFT-PROMPT-ID]
"""
import argparse
import os.path
import torch

from peft import PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from soft_prompt_model import PeftModelAdditionPrompt


def merge_lora(base_model_name, lora_path):

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    lora_model = PeftModel.from_pretrained(base_model, os.path.join(lora_path, 'global'))
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = lora_model.merge_and_unload()
    target_model_path = lora_path.replace("checkpoint", "full")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args.base_model_path, args.lora_path)