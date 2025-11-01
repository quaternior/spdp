S=$1
S_1=$2
S_2=$3
METHOD=$4
GPU_IDX=$5
eval_sparsity=$6

MODEL_TAG=llama2_7b/unstructured/${METHOD}${S_1}
MODEL_NAME=$SPDP_HOME/wanda/out/$MODEL_TAG
OUTPUT_PATH=$SPDP_HOME/TEAL/models/$MODEL_TAG/histogram

# To use general dense model, use this below. 
# MODEL_NAME=meta-llama/Llama-2-7b-hf
# OUTPUT_PATH=$SPDP_HOME/TEAL/models/$MODEL_NAME/histogram

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p $OUTPUT_PATH
fi

if [ "$7" = "grab" ]; then
    echo "grab_acts"

    CUDA_VISIBLE_DEVICES=$GPU_IDX python $SPDP_HOME/TEAL/teal/grab_acts.py --model_name $MODEL_NAME --output_path $OUTPUT_PATH
fi

if [ "$7" = "ppl" ]; then
    echo "PPL test"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDX python $SPDP_HOME/TEAL/teal/ppl_test.py --model_name $MODEL_NAME \
        --teal_path $OUTPUT_PATH \
        --sparsity $S_2"

    if [ "$eval_sparsity" = "True" ]; then
        CMD="$CMD --eval_sparsity"
    fi

    eval $CMD
fi

if [ "$7" = "grabnppl" ]; then
    echo "grab_acts"

    CUDA_VISIBLE_DEVICES=$GPU_IDX python $SPDP_HOME/TEAL/teal/grab_acts.py --model_name $MODEL_NAME --output_path $OUTPUT_PATH
    
    echo "PPL test"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_IDX python $SPDP_HOME/TEAL/teal/ppl_test.py --model_name $MODEL_NAME \
        --teal_path $OUTPUT_PATH \
        --sparsity $S_2"

    if [ "$eval_sparsity" = "True" ]; then
        CMD="$CMD --eval_sparsity"
    fi

    eval $CMD
fi