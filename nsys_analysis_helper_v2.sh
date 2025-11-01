
NSYS_PATH="$SPDP_HOME/../experiment_result/nsys_result/nsys_result/A10G"


ARGS="--disable-legend"
ARGS_S="--left-y-label"
ARGS_SPINFER="--left-y-label --s-list 0.45"
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spdp.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_from_csv_A10G.pdf" --s-start=0.3 $ARGS_S
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spinfer.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_spinfer_from_csv_A10G.pdf" --s-start=0.3 $ARGS_SPINFER
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_tpot_res_ppl_v2.py --results-dir $NSYS_PATH --output $NSYS_PATH/../tpot_vs_ppl_A10.pdf --overwrite $ARGS

# ################################3
NSYS_PATH="$SPDP_HOME/../experiment_result/nsys_result/nsys_result/L4"


ARGS="--disable-y-label --disable-legend"

ARGS_S="--legend"
ARGS_SPINFER="--legend"

python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spdp.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_from_csv_L4.pdf" --s-start=0.3 $ARGS_S
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spinfer.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_spinfer_from_csv_L4.pdf" --s-start=0.3 $ARGS_SPINFER
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_tpot_res_ppl_v2.py --results-dir $NSYS_PATH --output $NSYS_PATH/../tpot_vs_ppl_L4.pdf --overwrite $ARGS

##############################3

NSYS_PATH="$SPDP_HOME/../experiment_result/nsys_result/nsys_result/L40S"

ARGS="--disable-y-label"
ARGS_S="--right-y-label"

ARGS_SPINFER="--right-y-label"
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spdp.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_from_csv_L40S.pdf" --s-start=0.3 $ARGS_S
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_nsys_per_decoding_spinfer.py --results-dir $NSYS_PATH --output "$NSYS_PATH/plots/tpot_ppl_vs_sparsity_spinfer_from_csv_L40S.pdf" --s-start=0.3 $ARGS_SPINFER
python $SPDP_HOME/transformers-spdp/spdp_utils/etc/plot_tpot_res_ppl_v2.py --results-dir $NSYS_PATH --output $NSYS_PATH/../tpot_vs_ppl_L40S.pdf --overwrite $ARGS
