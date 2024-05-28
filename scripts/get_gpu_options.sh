#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PLATFORM=$($LOCAL_DIRECTORY/get_platform_name.sh)

if [ "$PLATFORM" == "rocm" ]; then
    GPU_OPTIONS="--network=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add=video \
        --ipc=host \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -it"
elif [ "$PLATFORM" == "nvidia" ]; then
    GPU_OPTIONS="--network=host \
        --device=/dev/dri \
        --gpus all"
elif [ "$PLATFORM" == "cpu" ]; then
    GPU_OPTIONS=""
else
    exit 1
fi

echo $GPU_OPTIONS