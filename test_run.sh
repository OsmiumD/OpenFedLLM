source setup.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/wsl/lib
CUDA_VISIBLE_DEVICES=1 python main_sft.py \
 --model_name_or_path "$HOME/models/Llama-2-7b-hf" \
 --dataset_name "vicgalle/alpaca-gpt4" \
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients 20 \
 --sample_clients 2 \
 --max_steps 10 \
 --num_rounds 200 \
 --batch_size 16 \
 --gradient_accumulation_steps 1 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --load_in_8bit \
 --output_dir "./output" \
 --template "alpaca"