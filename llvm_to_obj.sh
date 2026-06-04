#!/bin/bash

MLIR_TR=/home/thomas/repos/llvm-project/build/bin/mlir-translate
LLC=/home/thomas/repos/llvm-project/build/bin/llc

$MLIR_TR --mlir-to-llvmir $1 \
| $LLC -filetype=obj -mcpu=native -O3 -o $2
