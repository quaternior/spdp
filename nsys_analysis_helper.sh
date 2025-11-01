# for f in nsys_result/*.nsys-rep; do
#     base=$(basename "$f" .nsys-rep)
#     sqlite_path="nsys_result/$base.sqlite"
#     nohup nsys export -t sqlite -o "$sqlite_path" "$f" >/dev/null 2>&1 &
# done

# wait

# OUTPUT_TOKENS=256
# NSYS_PATH="$SPDP_HOME/../experiment_result/nsys_result/nsys_result/A10G"


# # Convert process
# for f in $NSYS_PATH/*.nsys-rep; do
#     base=$(basename "$f" .nsys-rep)
#     sqlite_path="$NSYS_PATH/${base}.sqlite"
#     log_path="$NSYS_PATH/${base}.log"
    
#     # Run conversion and log both stdout and stderr to .log file
#     nohup nsys export -f true -t sqlite -o "$sqlite_path" "$f" >"$log_path" 2>&1 &
# done

# wait
# echo "All exports completed. Logs are available in $NSYS_PATH/*.log"

# # Plot process
# for f in "$NSYS_PATH"/*.sqlite; do
#     base=$(basename "$f" .sqlite)
#     python "$SPDP_HOME/transformers-spdp/spdp_utils/etc/analyze_nsys_repeat.py" \
#         --db "$f" \
#         --nvtx-a "measurement range(Decoding)" \
#         --nvtx-b repeat \
#         --out "$NSYS_PATH/${base}.csv" \
#         --output-tokens "$OUTPUT_TOKENS"
# done

# # After analysis, you can plot using below

# Optionally control sparsity plotting range and interpolation targets via env vars:
#   export S_START=0.30
#   export S_END=0.60
#   export S_LIST="0.30,0.35,0.40"  # for SpInfer interpolation targets
# ARGS_S=""
# if [ -n "$S_START" ]; then ARGS_S="$ARGS_S --s-start $S_START"; fi
# if [ -n "$S_END" ]; then ARGS_S="$ARGS_S --s-end $S_END"; fi
# ARGS_SPINFER="$ARGS_S"
# if [ -n "$S_LIST" ]; then ARGS_SPINFER="$ARGS_SPINFER --s-list $S_LIST"; fi
# ARGS_SPINFER="--s-list 0.45"
ARGS_SPINFER="--s-list 0.45"
ARGS="--disable-legend"
# ARGS="--disable-y-label"
# ARGS="--disable-y-label --disable-legend"

# python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spdp.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_from_csv.png" --s-start=0.3 $ARGS_S
# python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spinfer.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_spinfer_from_csv.png" --s-start=0.3 $ARGS_SPINFER
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_tpot_res_ppl_v2.py --results-dir $NSYS_PATH --output $NSYS_PATH/plots/tpot_vs_ppl.png --overwrite $ARGS
# ###############################3

# NSYS_PATH="$SPDP_HOME/../experiment_result/nsys_result/nsys_result/L40S"
# # # Convert process
# # for f in $NSYS_PATH/*.nsys-rep; do
# #     base=$(basename "$f" .nsys-rep)
# #     sqlite_path="$NSYS_PATH/${base}.sqlite"
# #     log_path="$NSYS_PATH/${base}.log"
    
# #     # Run conversion and log both stdout and stderr to .log file
# #     nohup nsys export -f true -t sqlite -o "$sqlite_path" "$f" >"$log_path" 2>&1 &
# # done

# # wait
# # echo "All exports completed. Logs are available in $NSYS_PATH/*.log"

# # # Plot process
# # for f in "$NSYS_PATH"/*.sqlite; do
# #     base=$(basename "$f" .sqlite)
# #     python "$SPDP_HOME/transformers-spdp/spdp_utils/etc/analyze_nsys_repeat.py" \
# #         --db "$f" \
# #         --nvtx-a "measurement range(Decoding)" \
# #         --nvtx-b repeat \
# #         --out "$NSYS_PATH/${base}.csv" \
# #         --output-tokens "$OUTPUT_TOKENS"
# # done

# # # After analysis, you can plot using below

# # Optionally control sparsity plotting range and interpolation targets via env vars:
# #   export S_START=0.30
# #   export S_END=0.60
# #   export S_LIST="0.30,0.35,0.40"  # for SpInfer interpolation targets
# # ARGS_S=""
# # if [ -n "$S_START" ]; then ARGS_S="$ARGS_S --s-start $S_START"; fi
# # if [ -n "$S_END" ]; then ARGS_S="$ARGS_S --s-end $S_END"; fi
# # ARGS_SPINFER="$ARGS_S"
# # if [ -n "$S_LIST" ]; then ARGS_SPINFER="$ARGS_SPINFER --s-list $S_LIST"; fi
# # ARGS_SPINFER="--s-list 0.45"
# ARGS_SPINFER="--s-list 0.45"
# # ARGS="--disable-legend"
# ARGS="--disable-y-label"
# # ARGS="--disable-y-label --disable-legend"

# # python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spdp.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_from_csv.png" --s-start=0.3 $ARGS_S
# # python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spinfer.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_spinfer_from_csv.png" --s-start=0.3 $ARGS_SPINFER
# python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_tpot_res_ppl_v2.py --results-dir $NSYS_PATH --output $NSYS_PATH/plots/tpot_vs_ppl.png --overwrite $ARGS

# # python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_combined_from_csv.png"
# ################################3
# NSYS_PATH="$SPDP_HOME/../experiment_result/nsys_result/nsys_result/L4"

# # # Convert process
# # for f in $NSYS_PATH/*.nsys-rep; do
# #     base=$(basename "$f" .nsys-rep)
# #     sqlite_path="$NSYS_PATH/${base}.sqlite"
# #     log_path="$NSYS_PATH/${base}.log"
    
# #     # Run conversion and log both stdout and stderr to .log file
# #     nohup nsys export -f true -t sqlite -o "$sqlite_path" "$f" >"$log_path" 2>&1 &
# # done

# # wait
# # echo "All exports completed. Logs are available in $NSYS_PATH/*.log"

# # # Plot process
# # for f in "$NSYS_PATH"/*.sqlite; do
# #     base=$(basename "$f" .sqlite)
# #     python "$SPDP_HOME/transformers-spdp/spdp_utils/etc/analyze_nsys_repeat.py" \
# #         --db "$f" \
# #         --nvtx-a "measurement range(Decoding)" \
# #         --nvtx-b repeat \
# #         --out "$NSYS_PATH/${base}.csv" \
# #         --output-tokens "$OUTPUT_TOKENS"
# # done

# # # After analysis, you can plot using below

# # Optionally control sparsity plotting range and interpolation targets via env vars:
# #   export S_START=0.30
# #   export S_END=0.60
# #   export S_LIST="0.30,0.35,0.40"  # for SpInfer interpolation targets
# # ARGS_S=""
# # if [ -n "$S_START" ]; then ARGS_S="$ARGS_S --s-start $S_START"; fi
# # if [ -n "$S_END" ]; then ARGS_S="$ARGS_S --s-end $S_END"; fi
# # ARGS_SPINFER="$ARGS_S"
# # if [ -n "$S_LIST" ]; then ARGS_SPINFER="$ARGS_SPINFER --s-list $S_LIST"; fi
# # ARGS_SPINFER="--s-list 0.45"
# ARGS_SPINFER="--s-list 0.45"
# # ARGS="--disable-legend"
# # ARGS="--disable-y-label"
# ARGS="--disable-y-label --disable-legend"

# # python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spdp.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_from_csv.png" --s-start=0.3 $ARGS_S
# # python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spinfer.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_spinfer_from_csv.png" --s-start=0.3 $ARGS_SPINFER
# python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_tpot_res_ppl_v2.py --results-dir $NSYS_PATH --output $NSYS_PATH/plots/tpot_vs_ppl.png --overwrite $ARGS