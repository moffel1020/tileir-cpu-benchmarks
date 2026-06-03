#!/bin/bash

TTCPU_OPT=/home/thomas/repos/triton-cpu/build/cmake.linux-x86_64-cpython-3.10/bin/triton-opt

$TTCPU_OPT \
        --pass-pipeline='builtin.module(
      triton-cpu-scalarize{skip-gather-scatter=true},
      triton-cpu-convert-memory-ops{use-gather-scatter=true},
      triton-cpu-convert-ptr-ops,
      triton-cpu-convert-elementwise-ops,
      triton-cpu-convert-elem-manip-ops,
      triton-cpu-convert-dot-op,
      triton-cpu-convert-histogram-op,
      triton-cpu-convert-reduction{use-multidim-reduction-op=true use-reduction-op=false},
      triton-cpu-convert-scan,
      triton-cpu-convert-control-flow-op,
      triton-cpu-convert-atomic-ops,
      triton-cpu-convert-debug-ops,
      cse,
      symbol-dce,
      canonicalize,

      triton-cpu-canonicalize,
      triton-cpu-optimize-masks,
      canonicalize,
      triton-cpu-convert-dot-generic,
      triton-cpu-add-casts-for-unsupported-ops{promote-bf16-to-fp32=true convert-mixed-precision-matmul=true promote-lib-math-to-fp32=true},
      triton-cpu-decompose-fp-conversions{decompose-bf16-conversions=true decompose-fp8-conversions=true},
      cse,
      symbol-dce,
      canonicalize,

      tt.func(triton-cpu-lower-multi-reduction),
      expand-strided-metadata,
      convert-vector-to-scf{full-unroll=true target-rank=1 lower-tensors=false},
      lower-affine,
      convert-scf-to-cf,
      convert-index-to-llvm,
      triton-cpu-func-op-to-llvm,
      triton-cpu-get-program-id-op-to-llvm,
      triton-cpu-memory-op-to-llvm,
      triton-cpu-atomic-ops-to-llvm,
      triton-cpu-debug-ops-to-llvm,
      convert-math-to-llvm,
      convert-math-to-libm,
      convert-vector-to-llvm{reassociate-fp-reductions=true enable-x86=true vector-contract-lowering=outerproduct},
      finalize-memref-to-llvm,
      reconcile-unrealized-casts,
      convert-arith-to-llvm,
      convert-func-to-llvm,
      convert-ub-to-llvm,
      canonicalize,
      cse,
      symbol-dce,
      ensure-debug-info-scope-on-llvm-func
    )' \
    $1 -o $2
