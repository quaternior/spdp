# Specify output path to store activations and histograms
# MODEL_NAME=/root/MaskLLM/output/checkpoints/llama3.2_1b_hf_maskllm_c4/
MODEL_NAME=/root/wanda/out/llama2_7b/unstructured/wanda/
OUTPUT_PATH=./models/$MODEL_NAME/histogram

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p $OUTPUT_PATH
fi

CUDA_VISIBLE_DEVICES=0 python teal/grab_acts.py --model_name $MODEL_NAME --output_path $OUTPUT_PATH
