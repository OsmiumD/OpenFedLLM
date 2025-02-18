"""
Usage:
python convert_pt_adapter --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH] --soft_prompt [SOFT-PROMPT-ID]
"""
import argparse
import os.path
import torch

from peft import PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from soft_prompt_model import PeftModelAdditionPrompt


def merge_lora(base_model_name, lora_path):

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    client_model = PeftModel.from_pretrained(base_model, os.path.join(lora_path, 'client_specific'))

    target_model_path = os.path.join(lora_path, 'client_adapter')
    os.makedirs(target_model_path, exist_ok=True)
    client_pt_path = os.path.join(lora_path, 'clients')
    for client_pt in os.listdir(client_pt_path):
        pt = torch.load(os.path.join(client_pt_path, client_pt))
        set_peft_model_state_dict(client_model, pt)
        client_model.save_pretrained(os.path.join(target_model_path, client_pt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args.base_model_path, args.lora_path)