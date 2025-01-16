source env_setup.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/wsl/lib
python main_sft_prompt.py \
 --model_name_or_path "$MODELS_HOME/TinyLlama_v1.1" \
 --dataset_names "vicgalle/alpaca-gpt4" "TIGER-Lab/MathInstruct" "sahil2801/CodeAlpaca-20k"\
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients_dataset 2 \
 --sample_clients 3 \
 --max_steps 10 \
 --num_rounds 100 \
 --batch_size 16 \
 --gradient_accumulation_steps 1 \
 --seq_length 1024 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --balance_sample \
 --output_dir "./output" \
 --use_soft_prompt \
 --template "alpaca"
