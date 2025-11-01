git submodule update --init --recursive
source Init_spdp.sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
deactivate
conda deactivate
uv venv --clear
source .venv/bin/activate
uv pip install -r requirements.txt

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

hf auth login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# # To commit, use below.
# git checkout main