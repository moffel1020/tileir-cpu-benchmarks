#!/bin/bash

CUDA_TILE_OPT=/home/thomas/repos/cuda-tile-cpu/build/bin/cuda-tile-opt

$CUDA_TILE_OPT \
    --canonicalize --cse \
    --convert-cuda-tile-to-standard \
    --canonicalize --cse \
    --convert-elementwise-to-linalg \
    --linalg-fold-into-elementwise \
    --canonicalize --cse \
    --cuda-tile-cpu-tile-and-fuse-into-store="tile-sizes=32" \
    --canonicalize --cse \
    --eliminate-empty-tensors \
    --cuda-tile-cpu-vectorize-linalg \
    --canonicalize --cse \
    --lower-cuda-tile-cpu-mem-ops \
    --canonicalize --cse \
    --cuda-tile-cpu-insert-parallel-loops \
    --loop-invariant-code-motion \
    --loop-invariant-subset-hoisting \
    --canonicalize --cse \
    --eliminate-empty-tensors \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map" \
    --canonicalize --cse \
    --buffer-hoisting \
    --buffer-loop-hoisting \
    --promote-buffers-to-stack="max-alloc-size-in-bytes=128" \
    --canonicalize --cse \
    --buffer-deallocation-pipeline \
    --mem2reg \
    --sroa \
    --sccp \
    --canonicalize --cse \
    --lower-vector-multi-reduction="lowering-strategy=inner-reduction" \
    --convert-vector-to-scf \
    --canonicalize --cse \
    --lower-affine \
    --convert-scf-to-openmp \
    --convert-openmp-to-llvm \
    --convert-scf-to-cf \
    --expand-strided-metadata \
    --finalize-memref-to-llvm \
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