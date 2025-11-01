# Dependency:
# Environment should activated and requirements are installed
# Init_spdp set
# S_1_SET=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8)
S_1_SET=(0.4 0.5 0.6 0.7 0.8)
FORMAT=spinfer

S_1_PREV=0
for ((i=0; i<${#S_1_SET[@]}; i++)); do
    S_1=${S_1_SET[i]}
    deactivate
    source $SPDP_HOME/.venv/bin/activate
    
    # 3. Edit model(If not cached)
    # cd $SPDP_HOME/transformers-spdp
    # python -m spdp_utils.scripts.run_edit_model \
    #     --cached \
    #     --repo $SPDP_HOME/spdp/wanda/out/llama2_7b/unstructured/wanda${S_1} \
    #     --target_repo meta-llama/llama-7b-hf-wanda${S_1} \
    #     --cache_dir $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda${S_1} \
    #     --out_dir  $SPDP_HOME/transformers-spdp/spdp_utils/
    # Clean cache
    rm -rf "$SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-2-7b-hf-wanda${S_1_PREV}-spinfer"
    rm -rf $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda${S_1_PREV}

    # 2. Static pruning(If not cached)
    cd $SPDP_HOME/wanda/
    source $SPDP_HOME/wanda/wanda.sh $S_1 $S_1 wanda 0
    
    # 3. Edit model
    cd $SPDP_HOME/transformers-spdp
    python -m spdp_utils.scripts.run_edit_model --format spinfer \
        --cached \
        --repo $SPDP_HOME/spdp/wanda/out/llama2_7b/unstructured/wanda${S_1} \
        --target_repo meta-llama/llama-2-7b-hf-wanda${S_1}-spinfer \
        --cache_dir $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda${S_1} \
        --out_dir  $SPDP_HOME/transformers-spdp/spdp_utils/
        
    # 4. Run model
    cd $SPDP_HOME/transformers-spdp/spdp_utils/scripts/
    # python run_sparse_model.py \
    #   --model-id $SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-7b-hf-wanda${S_1} \
    #   --device cuda --dtype float16 \
    #   --attn sdpa \
    #   --prompt "You are a very helpful assistant. Answer concisely.\n User : Please explain about what attention is in the transformers.\n Assistant : " \
    #   --max-new-tokens 256 \
    #   --warmup 5 --repeat 15 \
    #   --save-per-iter 1 \
    #   --outdir $SPDP_HOME/transformers-spdp/spdp_utils/results \
    #   --dynamic-pruning True \
    #   --histogram-path $SPDP_HOME/TEAL/models/llama2_7b/unstructured/wanda${S_1}/histogram \
    #   --dp-ratio $S_1 \
    #   --sp-ratio $S_2 \
    #   --compression-format $FORMAT

    python run_model.py \
        --model-id $SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-2-7b-hf-wanda${S_1}-spinfer \
        --device cuda --dtype float16 \
        --attn flash_attention_2 \
        --input-tokens 32 \
        --max-new-tokens 1280 \
        --warmup 5 --repeat 15 \
        --save-per-iter 1 \
        --outdir $SPDP_HOME/transformers-spdp/spdp_utils/results \
        --sp-ratio $S_1 \
        --compression-format $FORMAT

    S_1_PREV=$S_1
done