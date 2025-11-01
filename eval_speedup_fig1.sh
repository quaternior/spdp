# Dependency:
# Environment should activated and requirements are installed
# Init_spdp set
# S_1_SET=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8)
S_1_SET=(0.5 0.55 0.6 0.65 0.7 0.75 0.8)

FORMAT=SPDP
TEAL_DEPTH=grab

S_1_PREV=0
for ((i=0; i<${#S_1_SET[@]}; i++)); do
    S_1=${S_1_SET[i]}
    S_2=${S_2_SET[i]}

    source $SPDP_HOME/.venv/bin/activate
    if [[ "$S_1" != "$S_1_PREV" ]]; then
        rm -rf "$SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-7b-hf-wanda${S_1_PREV}"
        rm -rf $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda${S_1_PREV}
        # 1. Static pruning(If not cached)
        cd $SPDP_HOME/wanda/
        source $SPDP_HOME/wanda/wanda.sh $S_1 $S_1 wanda 0

        # 2. Dynamic pruning(collect histogram), only for SPDP if not cached
        
        # if [ "$FORMAT" = "SPDP" ]; then
        #     source $SPDP_HOME/TEAL/scripts/grabnppl.bash -1 $S_1 0 wanda 0 False $TEAL_DEPTH
        # fi
    fi
    ######################################################################################
    # [DEBUG] logging for PPL
    TEAL_DEPTH_DEBUG=ppl
    source $SPDP_HOME/TEAL/scripts/grabnppl.bash -1 $S_1 0 wanda 0 False $TEAL_DEPTH_DEBUG
    ######################################################################################
    S_1_PREV=$S_1
done