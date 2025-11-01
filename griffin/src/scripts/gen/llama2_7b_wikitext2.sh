model=llama2
# Select index 0(7b)
modelsize=0
dataset=wikitext
# PPL, zero-shot
shots=0
mode=ppl
device=0

# Full
# python ppl_test.py \
#     --tasks $dataset \
#     --num_fewshot $shots \
#     --model_arch $model \
#     --model_size $modelsize \
#     --density 1 \
#     --device cuda:$device \ 
#     --limit 250

# # Magnitude
# python lm_eval.py \
#     --tasks $dataset \
#     --num_fewshot $shots \
#     --model_arch $model \
#     --model_size $modelsize \
#     --density 0.5 \
#     --mode $mode \
#     --selection_method magnitude \
#     --device cuda:$device

# GRIFFIN
python ppl_test.py \
    --tasks $dataset \
    --num_fewshot $shots \
    --limit 250 \
    --model_arch $model \
    --model_size $modelsize \
    --density 0.5 \
    --mode $mode \
    --selection_method topk \
    --device cuda:$device
