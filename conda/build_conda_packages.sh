#!/bin/bash

set -o xtrace
set -e

if [ -z $PYVER ]; then
    echo "PYVER is not set"
    exit 1
fi

export CUDA_MAJOR=10
export CUDA_MINOR=2
export CUDA_VERSION=$CUDA_MAJOR$CUDA_MINOR

# Adding conda-forge channel for dependencies
conda config --add channels nvidia
conda config --add channels conda-forge

CONDA_BUILD_OPTIONS="--python=${PYVER} --exclusive-config-file config/conda_build_config.yaml"

CONDA_PREFIX=${CONDA_PREFIX:-/root/miniconda3}

export DALI_CONDA_BUILD_VERSION=$(cat ../VERSION)$(if [ "${NVIDIA_DALI_BUILD_FLAVOR}" != "" ]; then \
                                                     echo .${NVIDIA_DALI_BUILD_FLAVOR}.${DALI_TIMESTAMP}; \
                                                   fi)
# Build custom OpenCV first, as DALI requires only bare OpenCV without many features and dependencies
conda build ${CONDA_BUILD_OPTIONS} third_party/dali_opencv/recipe

# Build custom FFmpeg, as DALI requires only bare FFmpeg without many features and dependencies
# but wiht mpeg4_unpack_bframes enabled
conda build ${CONDA_BUILD_OPTIONS} third_party/dali_ffmpeg/recipe

# Building DALI package
conda build ${CONDA_BUILD_OPTIONS} recipe

# Copying the artifacts from conda prefix
mkdir -p artifacts
cp ${CONDA_PREFIX}/conda-bld/*/nvidia-dali*.tar.bz2 artifacts
