#!/bin/bash

MLIR_TR=../llvm-project/build/bin/mlir-translate
LLC=../llvm-project/build/bin/llc

$MLIR_TR --mlir-to-llvmir $1 \
| $LLC -filetype=obj -mcpu=native -O3 -o $2
