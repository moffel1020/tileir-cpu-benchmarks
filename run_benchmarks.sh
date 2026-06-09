#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

launchers=(
  "build/launchers/layernorm_launch_cpp"
  "build/launchers/layernorm_launch_triton"
  "build/launchers/layernorm_launch_cutile"
  "build/launchers/resize_launch_cpp"
  "build/launchers/resize_launch_triton"
  "build/launchers/resize_launch_cutile"
  "build/launchers/sharpen_launch_cpp"
  "build/launchers/sharpen_launch_triton"
  "build/launchers/sharpen_launch_cutile"
  "build/launchers/softmax_launch_cpp"
  "build/launchers/softmax_launch_triton"
  "build/launchers/softmax_launch_cutile"
  "build/launchers/swiglu_launch_cpp"
  "build/launchers/swiglu_launch_triton"
  "build/launchers/swiglu_launch_cutile"
  "build/launchers/warp_launch_cpp"
  "build/launchers/warp_launch_triton"
  "build/launchers/warp_launch_cutile"
)

thread_counts=(1 2 4 6 8 10 12)

for threads in "${thread_counts[@]}"; do
  export OMP_NUM_THREADS="$threads"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

  for launcher in "${launchers[@]}"; do
    echo "$launcher"
    "$launcher"
  done
done
