# Dependency:
# Environment should activated and requirements are installed
# Init_spdp set
S_1_SET=(0.05 0.05 0.05 0.05 0.05 0.1 0.1 0.15 0.2 0.25 0.25 0.3 0.35 0.4 0.45)
S_2_SET=(0.053 0.105 0.158 0.211 0.263 0.278 0.333 0.353 0.375 0.4 0.467 0.5 0.538 0.583 0.636)
FORMAT=SPDP
TEAL_DEPTH=grab

S_1_PREV=0
for ((i=0; i<${#S_1_SET[@]}; i++)); do
    S_1=${S_1_SET[i]}
    S_2=${S_2_SET[i]}

    source $SPDP_HOME/.venv/bin/activate
    if [[ "$S_1" != "$S_1_PREV" ]]; then
        rm -rf "$SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-2-7b-hf-wanda${S_1_PREV}"
        rm -rf $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda${S_1_PREV}
        # 1. Static pruning(If not cached)
        cd $SPDP_HOME/wanda/
        source $SPDP_HOME/wanda/wanda.sh $S_1 $S_1 wanda 0

        # 2. Dynamic pruning(collect histogram), only for SPDP if not cached
        
        if [ "$FORMAT" = "SPDP" ]; then
            source $SPDP_HOME/TEAL/scripts/grabnppl.bash -1 $S_1 $S_2 wanda 0 False $TEAL_DEPTH
        fi
    fi
    ######################################################################################
    # # [DEBUG] logging for PPL
    # TEAL_DEPTH_DEBUG=ppl
    # source $SPDP_HOME/TEAL/scripts/grabnppl.bash -1 $S_1 $S_2 wanda 0 False $TEAL_DEPTH_DEBUG
    ######################################################################################
    
    # 3. Edit model(If not cached)
    if [[ "$S_1" != "$S_1_PREV" ]]; then
        cd $SPDP_HOME/transformers-spdp
        python -m spdp_utils.scripts.run_edit_model \
          --cached \
          --repo $SPDP_HOME/spdp/wanda/out/llama2_7b/unstructured/wanda${S_1} \
          --target_repo meta-llama/llama-2-7b-hf-wanda${S_1} \
          --cache_dir $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda${S_1} \
          --out_dir  $SPDP_HOME/transformers-spdp/spdp_utils/
    fi

    # 4. Run model
    cd $SPDP_HOME/transformers-spdp/spdp_utils/scripts/
    python run_sparse_model.py \
      --model-id $SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-2-7b-hf-wanda${S_1} \
      --device cuda --dtype float16 \
      --attn flash_attention_2 \
      --prompt "You are a very helpful assistant. Answer concisely.\n User : Please explain about what attention is in the transformers.\n Assistant : " \
      --input-tokens 32 \
      --max-new-tokens 1280 \
      --warmup 5 --repeat 15 \
      --save-per-iter 1 \
      --outdir $SPDP_HOME/transformers-spdp/spdp_utils/results \
      --dynamic-pruning True \
      --histogram-path $SPDP_HOME/TEAL/models/llama2_7b/unstructured/wanda${S_1}/histogram \
      --dp-ratio $S_1 \
      --sp-ratio $S_2 \
      --compression-format $FORMAT
    
    S_1_PREV=$S_1
done