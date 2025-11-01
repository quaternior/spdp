# TEAL_PATH is the output path specified in grab_acts.py
# MODEL_NAME=/root/MaskLLM/output/checkpoints/llama2_7b_hf_wanda_c4/
# MODEL_NAME=/root/wanda/out/llama2_7b/unstructured/wanda/
MODEL_NAME=meta-llama/Llama-2-7b-hf

OUTPUT_PATH=./models/$MODEL_NAME/histogram
SPARSITY=0.75

CUDA_VISIBLE_DEVICES=1 python teal/ppl_test.py --model_name $MODEL_NAME --teal_path $OUTPUT_PATH --sparsity $SPARSITY


