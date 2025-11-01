S=$1

MODEL_NAME=/root/wanda/out/llama2_7b/unstructured/wanda${S}
OUTPUT_PATH=/root/TEAL/models/$MODEL_NAME/histogram

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p $OUTPUT_PATH
fi

CUDA_VISIBLE_DEVICES=0 python teal/ppl_test_woteal.py --model_name $MODEL_NAME --teal_path $OUTPUT_PATH
