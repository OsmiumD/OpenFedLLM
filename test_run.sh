source env_setup.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/wsl/lib
python main_sft.py \
 --model_name_or_path "$MODELS_HOME/TinyLlama_v1.1" \
 --dataset_name "TIGER-Lab/MathInstruct" \
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients 1 \
 --sample_clients 1 \
 --max_steps 10 \
 --num_rounds 200 \
 --batch_size 16 \
 --gradient_accumulation_steps 1 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --output_dir "./output" \
 --template "alpaca"

# TIGER-Lab/MathInstruct
# vicgalle/alpaca-gpt4
