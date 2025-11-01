S=$1
eval_sparsity=$2

MODEL_NAME=meta-llama/Llama-2-7b-hf
OUTPUT_PATH=/root/TEAL/models/$MODEL_NAME/histogram

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p $OUTPUT_PATH
fi

CMD = "CUDA_VISIBLE_DEVICES=0 python teal/ppl_test.py --model_name $MODEL_NAME --teal_path $OUTPUT_PATH --sparsity $S"

if [ "$eval_sparsity" = "True" ]; then
    CMD="$CMD --eval_sparsity"
fi

eval CMD

