#!/bin/bash

MLIR_TR=$LLVM_BUILD_DIR/bin/mlir-translate
OPT=$LLVM_BUILD_DIR/bin/opt
LLC=$LLVM_BUILD_DIR/bin/llc

$MLIR_TR --mlir-to-llvmir $1 \
| $OPT -O3 \
| $LLC -filetype=obj -mcpu=native -O3 -o $2
