#!/bin/bash
module load CUDA/9.2.88
export TOOLS_HOME=$HOME/py/install
#export UCX_HOME=$HOME/ucx-1.5.0/build
export UCX_HOME=$HOME/ucx-github/build
export CUDA_HOME=/gpfs/sw/software/CUDA/9.2.88
export PATH=$TOOLS_HOME/bin:$PATH
export PATH=$UCX_HOME/bin:$PATH
export LD_LIBRARY_PATH=$TOOLS_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/gpfs/sw/gdrdrv/gdrcopy:$LD_LIBRARY_PATH
export UCX_PY_CUDA_PATH=$CUDA_HOME
export UCX_PY_UCX_PATH=$UCX_HOME
#export PYTHONPATH=$HOME/tools/install/lib/python3.7:$PYTHONPATH
#export NUMBAPRO_NVVM=/gpfs/sw/software/CUDA/9.2.88/nvvm/lib64/libnvvm.so
#export NUMBAPRO_CUDA_DRIVER=/usr/lib64/libcuda.so
#export NUMBAPRO_LIBDEVICE=/gpfs/sw/software/CUDA/9.2.88/nvvm/libdevice
