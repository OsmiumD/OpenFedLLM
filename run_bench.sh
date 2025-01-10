source ./env_setup.sh

# model directory setup
base_model_name=TinyLlama_v1.1
output_name=alpaca-gpt4_20000_fedavg_c20s2_i10_b16a1_l512_r32a64_20250108125314
ckpt_name=checkpoint-200
lora_path=$OUTPUT_DIR/$output_name/$ckpt_name
full_name=${ckpt_name/checkpoint/full}
# merge lora
# python ./utils/merge_lora.py --base_model_path $MODELS_HOME/$base_model_name\
# --lora_path ./output/$output_name/$ckpt_name

#gen_answer_mt
cd ./evaluation/open_ended || exit
python gen_model_answer_mt.py \
    --base_model_path "$MODELS_HOME/$base_model_name"\
    --template alpaca \
    --lora_path "$lora_path"