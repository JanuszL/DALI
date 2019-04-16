#!/bin/bash -e
# used pip packages

pip_packages="tensorflow-gpu torchvision mxnet-cu##CUDA_VERSION##"
one_config_only=true

pushd ../..

cd dali/test/python

NUM_GPUS=$(nvidia-smi -L | wc -l)

test_body() {
    python test_RN50_data_fw_iterators.py --gpus ${NUM_GPUS} -b 13 --workers 3 --prefetch 2 -i 100 --epochs 2
}

source ../../../qa/test_template.sh

popd
