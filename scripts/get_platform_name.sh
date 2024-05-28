#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

if ! command -v rocm-smi &> /dev/null
then

    if ! command -v nvidia-smi &> /dev/null
    then
        echo "cpu"
        exit
    fi

    echo "nvidia"
    exit
fi

echo "rocm"