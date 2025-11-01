S=$1
S_1=$2
METHOD=$3
GPU_IDX=$4
CUDA_VISIBLE_DEVICES=$GPU_IDX python -u $SPDP_HOME/wanda/main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method ${METHOD} \
    --sparsity_ratio $S_1 \
    --sparsity_type unstructured \
    --save $SPDP_HOME/wanda/out/llama2_7b/unstructured/${METHOD}${S_1}_log \
    --save_model $SPDP_HOME/wanda/out/llama2_7b/unstructured/${METHOD}${S_1} \
    --cache_dir $SPDP_HOME/llm_weights/