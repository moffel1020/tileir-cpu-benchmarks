#!/bin/bash

CUDA_TILE_OPT=/home/thomas/repos/cuda-tile-cpu/build/bin/cuda-tile-opt

$CUDA_TILE_OPT \
    --canonicalize --cse \
    --convert-cuda-tile-to-standard \
    --canonicalize --cse \
    --convert-elementwise-to-linalg \
    --linalg-fold-into-elementwise \
    --canonicalize --cse \
    --eliminate-empty-tensors \
    --cuda-tile-cpu-vectorize-linalg \
    --canonicalize --cse \
    --lower-cuda-tile-cpu-mem-ops \
    --canonicalize --cse \
    --cuda-tile-cpu-insert-parallel-loops \
    --loop-invariant-code-motion \
    --loop-invariant-subset-hoisting \
    --eliminate-empty-tensors \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
    --canonicalize --cse \
    --mem2reg \
    --canonicalize --cse \
    --buffer-hoisting \
    --buffer-loop-hoisting \
    --promote-buffers-to-stack \
    --mem2reg \
    --sroa \
    --sccp \
    --canonicalize --cse \
    --convert-scf-to-openmp \
    --expand-strided-metadata \
    --finalize-memref-to-llvm \
    --lower-vector-multi-reduction="lowering-strategy=inner-reduction" \
    --convert-vector-to-scf \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-vector-to-llvm="reassociate-fp-reductions=true enable-x86=true vector-contract-lowering=outerproduct" \
    --convert-cuda-tile-cpu-to-llvm \
    --convert-to-llvm \
    --reconcile-unrealized-casts \
    --canonicalize --cse \
    --remove-dead-values \
    $1 -o $2

#     --cuda-tile-cpu-tile-and-fuse-into-store="tile-sizes=16" \
    # --linalg-fuse-elementwise-ops \ removed for layernorm

# after fold elemwise ops:
    # --canonicalize --cse \
    # --linalg-fuse-elementwise-ops \
    # --canonicalize --cse \
    # --cuda-tile-cpu-tile-and-fuse-into-store="tile-sizes=16" \