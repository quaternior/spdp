The overall initialization, submodule and installation can be installed in one shot as below:
```
source install_one_shot.sh
```
# Installation
Initialize your path first. 
```
source Init_spdp.sh
```
Install uv.
```
source Init_spdp.sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

# Finding optimal sparsity ratio
## 1. Wanda & TEAL Install

Step 1: Wanda & TEAL Install
```
uv venv .venv
source .venv/bin/activate
cd $SPDP_HOME/TEAL
pip install -e .

pip install -e ".[eval]" # For evaluating our models
```
You must login on huggingface to use Llama-2-7b-hf model.

Step 1: Evaluate run pruning
```
cd $SPDP_HOME/TEAL
python run_pruning.py
```
With the form [s1, s2, PPL], csv file will be saved in `TEAL/results`

# End-to-end kernel performance
See extension-pytorch repository(https://github.com/quaternior/extension-pytorch) to use pytorch bindings, SPDP-spMspV/SpMM kernels in csrc/spmm-kernel directory


# End-to-end LLM inference
## 1. Installation & Init
```
git submodule update --init --recursive

cd $SPDP_HOME && uv pip install -r requirements.txt
# Install SPDP_KERNEL
cd $SPDP_HOME/extension-pytorch/

cd csrc/spdp-kernel && source Init_SPDP_kernel.sh
cd kernel_benchmark && source test_env
source make_all.sh

cd $SPDP_HOME/extension-pytorch
uv pip install --no-build-isolation -e .

# Install hf transformer binded w/ SPDP_KERNEL
cd $SPDP_HOME/transformers-spdp/
uv pip install -e .[torch]

# Install TEAL
cd $SPDP_HOME/TEAL
uv pip install -e .
uv pip install -e ".[eval]" --no-build-isolation
```
## 2. Model edit & LLM inference test
See transformers-spdp repository(https://github.com/quaternior/transformers-spdp) to edit model and inference LLM