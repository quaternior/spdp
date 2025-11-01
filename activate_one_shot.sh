git submodule update --init --recursive

conda deactivate
deactivate

source Init_spdp.sh

# Install SPDP_KERNEL
cd $SPDP_HOME/extension-pytorch/

cd csrc/spdp-kernel && source Init_SPDP_kernel.sh
cd kernel_benchmark && source test_env

# Install SPDP
cd $SPDP_HOME/

source .venv/bin/activate
