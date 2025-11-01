# SPDP: Sparse LLM Inference Stack

This repository provides an end-to-end inference pipeline that combines activation sparsity (TEAL) and weight sparsity (Wanda), accelerated by custom sparse kernels (SPDP/SpInfer). Kernels are built as a PyTorch extension and integrated into a Hugging Face Transformers fork (transformers‑spdp) to measure TTFT/TPOT performance and optionally profile with NVIDIA Nsight Systems.

- Static pruning: generate weight sparsity with Wanda
- Dynamic pruning: use TEAL to calibrate activation thresholds / collect histograms and evaluate PPL
- Model conversion: compress weights (projections) into SPDP or SpInfer formats and update config files
- Inference/benchmark: repeatedly measure TTFT/TPOT, choose Flash‑Attention 2 or SDPA, and save results as JSON
- Kernels: call spMspV/SpMM kernels via PyTorch bindings and run standalone kernel benchmarks

> Prerequisites
> - CUDA GPU environment; NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer); Python 3.11 recommended; PyTorch 2.8.0 (pinned)
> - Access to meta‑llama/Llama‑2‑7b‑hf and a Hugging Face token
> - Optional: NVIDIA Nsight Systems (nsys) for profiling


## Scope and Reproducibility

- Goals
  - Build and validate the sparse kernel PyTorch extension.
  - Convert models into SPDP/SpInfer formats and run end‑to‑end inference.
  - Measure TTFT/TPOT across dynamic (SPDP) and static (SpInfer) sparsity regimes.
  - Optionally profile per‑token latency using Nsight Systems and generate plots.
- What you can reproduce
  - PPL vs sparsity sweeps using TEAL + Wanda.
  - End‑to‑end timing runs producing JSON outputs.
  - NSYS‑based per‑token breakdowns and summary plots.
- Notes on variance
  - Timing varies with GPU, drivers, and background load. The scripts use warmup/repeat loops; keep the machine as idle as possible for stable numbers.
  - Some steps cache intermediate artifacts; re‑runs are faster and more consistent.


## Quick Start (one‑shot install)

The script initializes submodules, creates a uv/virtual environment, builds kernels, installs dependencies, and logs into Hugging Face in one go.

```
export HUGGINGFACE_TOKEN=...   # token with access to Llama‑2
source install_one_shot.sh
```

To re‑activate the environment in a new shell:

```
source activate_one_shot.sh
```


## Sanity Checks

After setup, validate the installation with a lightweight sequence:

- Kernel binding unit test
```
python extension-pytorch/test/test_spmm.py
```

- Minimal end‑to‑end run (dynamic pruning path)
  - Prerequisite: run the model conversion step in “A. SPDP (dynamic) route → 1) Model conversion” to create the local `llama-2-7b-hf-wanda0.5` folder.
```
cd $SPDP_HOME/transformers-spdp/spdp_utils/scripts
python run_spdp_model.py \
  --model-id $SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-2-7b-hf-wanda0.5 \
  --device cuda --dtype float16 \
  --attn flash_attention_2 \
  --input-tokens 32 \
  --max-new-tokens 64 \
  --warmup 2 --repeat 3 \
  --save-per-iter 1 \
  --outdir $SPDP_HOME/transformers-spdp/spdp_utils/results \
  --dynamic-pruning True \
  --histogram-path $SPDP_HOME/TEAL/models/llama2_7b/unstructured/wanda0.5/histogram \
  --dp-ratio 0.5 \
  --sp-ratio 0.53 \
  --compression-format SPDP
```


## Manual Installation (detailed)

1) Initialize paths and submodules
```
source Init_spdp.sh               # sets SPDP_HOME
git submodule update --init --recursive
```

2) uv/virtualenv + common dependencies
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

3) Build kernels and install PyTorch bindings
```
cd $SPDP_HOME/extension-pytorch/
cd csrc/spdp-kernel && source Init_SPDP_kernel.sh
cd kernel_benchmark && source test_env && source make_all.sh

cd $SPDP_HOME/extension-pytorch
uv pip install --no-build-isolation -e .
```

4) Install the Transformers (SPDP fork)
```
cd $SPDP_HOME/transformers-spdp
uv pip install -e .[torch]
```

5) Install TEAL (including eval)
```
cd $SPDP_HOME/TEAL
uv pip install -e .
uv pip install -e ".[eval]" --no-build-isolation
```

6) Hugging Face authentication (to use Llama‑2 weights)
```
hf auth login --token $HUGGINGFACE_TOKEN --add-to-git-credential
```


## Search the optimal sparsity ratio (TEAL + Wanda)

Use Wanda to produce static pruning artifacts, and TEAL to collect activation histograms, calibrate thresholds, and evaluate PPL.

1) Verify TEAL installation and run:
```
cd $SPDP_HOME/TEAL
python run_pruning.py
```

When the run finishes, a CSV with columns `s1, s2, Uniform_PPL` is created under `TEAL/results/`.
- s1: dynamic pruning ratio (DP)
- s2: static sparsity ratio aligned to the final model sparsity (SP)

Expected outputs
- CSV: `TEAL/results/*.csv` with columns `s1, s2, Uniform_PPL`.
- Histograms: under `TEAL/models/llama2_7b/unstructured/wanda<s1>/histogram` when using the provided scripts.


## End‑to‑end LLM inference

Two routes are provided: (A) SPDP dynamic pruning, (B) SpInfer static (compressed) inference.

### A. SPDP (dynamic) route

1) Model conversion (example: Wanda 50%)
```
cd $SPDP_HOME/transformers-spdp
python -m spdp_utils.scripts.run_edit_model --format spdp \
  --cached \
  --repo $SPDP_HOME/spdp/wanda/out/llama2_7b/unstructured/wanda0.5 \
  --target_repo meta-llama/llama-2-7b-hf-wanda0.5 \
  --cache_dir $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda0.5 \
  --out_dir  $SPDP_HOME/transformers-spdp/spdp_utils/
```

2) Dynamic pruning inference/measurement 
```
cd $SPDP_HOME/transformers-spdp/spdp_utils/scripts
python run_spdp_model.py \
  --model-id $SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-2-7b-hf-wanda0.5 \
  --device cuda --dtype float16 \
  --attn flash_attention_2 \
  --input-tokens 32 \
  --max-new-tokens 256 \
  --warmup 5 --repeat 15 \
  --save-per-iter 1 \
  --outdir $SPDP_HOME/transformers-spdp/spdp_utils/results \
  --dynamic-pruning True \
  --histogram-path $SPDP_HOME/TEAL/models/llama2_7b/unstructured/wanda0.5/histogram \
  --dp-ratio 0.5 \
  --sp-ratio 0.53 \
  --compression-format SPDP
```

Result JSONs are saved under `transformers-spdp/spdp_utils/scripts/results/`.

### B. SpInfer (static) route

1) Model conversion (SpInfer format)
```
cd $SPDP_HOME/transformers-spdp
python -m spdp_utils.scripts.run_edit_model --format spinfer \
  --cached \
  --repo $SPDP_HOME/spdp/wanda/out/llama2_7b/unstructured/wanda0.6 \
  --target_repo meta-llama/llama-2-7b-hf-wanda0.6-spinfer \
  --cache_dir $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda0.6 \
  --out_dir  $SPDP_HOME/transformers-spdp/spdp_utils/
```

2) Static sparse inference/measurement 
```
cd $SPDP_HOME/transformers-spdp/spdp_utils/scripts
python run_sparse_model.py \
  --model-id $SPDP_HOME/transformers-spdp/spdp_utils/meta-llama/llama-2-7b-hf-wanda0.6-spinfer \
  --device cuda --dtype float16 \
  --attn flash_attention_2 \
  --input-tokens 32 \
  --max-new-tokens 256 \
  --warmup 5 --repeat 15 \
  --save-per-iter 1 \
  --outdir $SPDP_HOME/transformers-spdp/spdp_utils/results \
  --sp-ratio 0.6 \
  --compression-format spinfer
```


## Kernel benchmarks

PyTorch bindings and kernel sources live in `extension-pytorch`. See the sub-README for details and test instructions.
- Repository: https://github.com/quaternior/extension-pytorch
- Kernel path: `extension-pytorch/csrc/spdp-kernel`
- Quick test: `python extension-pytorch/test/test_spmm.py`


## NSYS profiling/analysis (optional)

You can profile with NSYS using NVTX ranges to separate per‑token sections.

```
cd $SPDP_HOME/transformers-spdp/spdp_utils/scripts
nsys profile -f true --cuda-event-trace=false -o $SPDP_HOME/e2e_spdp \
  python run_spdp_model.py ...
```

Example scripts to transform/plot results:
```
# Generate CSV/figures (edit paths for your env)
bash $SPDP_HOME/nsys_analysis_helper_v2.sh
```

Expected outputs
- Nsight Systems report: created with the `-o` prefix you choose (e.g., `$SPDP_HOME/e2e_spdp.*`).
- Plots from helper: PDFs such as `tpot_ppl_vs_sparsity_from_csv_*.pdf` in a `plots/` subfolder of the configured results dir.


## Script summary

- `install_one_shot.sh`: one‑shot setup of submodules/kernels/deps/HF auth
- `activate_one_shot.sh`: initialize kernel env, then activate the virtualenv
- `eval_speedup*.sh`: repeat convert/inference over various (DP, SP) combos
- `nsys_analysis_helper*.sh`: transform nsys outputs and plot speed–accuracy
- `tools/model_upload.py`: upload local conversion results to the HF Hub


## Folder structure

- `extension-pytorch`: SPDP kernels (PyTorch bindings, build/test scripts)
- `transformers-spdp`: HF Transformers fork + `spdp_utils` (convert/bench scripts)
- `TEAL`: activation sparsity (histogram collection, threshold calibration, PPL eval)
- `wanda`: static pruning tools and wrapper scripts
- `tools`: supporting utilities (e.g., HF upload)
- `eval_speedup*.sh`: experiment scripts for reproducibility


## Notes

- Always set `SPDP_HOME` via `source Init_spdp.sh`.
- Llama‑2 weights require a HF account permission and token.
- When using Flash‑Attention 2, verify GPU/driver compatibility.
- If the kernel/Triton environment is not prepared, sparse paths may automatically fall back to dense.


## Reproduction Guide

The following scripted paths automate the sweeps used in our evaluations. Ensure `SPDP_HOME` is set and the environment is activated.

- Dynamic pruning sweeps (SPDP)
```
bash eval_speedup.sh
```
  - Iterates over `(DP=s1, SP=s2)` pairs, generates/edit models, and runs inference.
  - Outputs JSON timing results under `transformers-spdp/spdp_utils/results/` and uses TEAL histograms under `TEAL/models/.../histogram` when present.

- Static sparsity sweeps (SpInfer)
```
bash eval_speedup_spinfer.sh
```
  - Converts to SpInfer format and runs static sparse inference across sparsity levels.

- Timing with Nsight Systems
```
bash eval_speedup_nsys.sh
```
  - Same as dynamic sweeps but intended for NVTX/NSYS collection.


## Determinism and Measurement

- Timing numbers are subject to variance; use the provided `--warmup` and `--repeat` settings and minimize other system activity.
- Model conversion and pruning steps are cached by default (`--cached`); delete the relevant folders under `wanda/out/...` and `transformers-spdp/spdp_utils/meta-llama/...` to force regeneration.
- If encountering CUDA OOM, reduce `--max-new-tokens` or the batch/input size.


## References

- TEAL: Training‑Free Activation Sparsity in LLMs — https://github.com/FasterDecoding/TEAL
- Transformers (SPDP fork): https://github.com/quaternior/transformers-spdp
