# Tile IR CPU benchmarks

# Running benchmarks
First built cuda-tile-cpu and triton-cpu. After set the following ENV variables:
```
LLVM_BUILD_DIR=/path/to/llvm_build_dir/
TTCPU_OPT=/path/to/triton-opt
CUDA_TILE_OPT=/path/to/cuda-tile-opt
```

Then run the following commands:
```
make cutile
make triton
make launchers
./run_benchmarks
```
These first commands first lower the CUDA Tile IR and Triton IR to the LLVM dialect, and then compiles them using `llc`.

# Exporting kernels to mlir
The provided python kernels are already exported to mlir inside the `kernels_ir/` directory. 
To re-export them (if they are modified) do the following:

## cuda tile
First install cutile-python.
then `python3 export_cutile_kernels`

## triton cpu
Make sure pytorch (cpu), numpy and triton-cpu are installed for python.
then `python3 export_triton_kernels`

# Acknowledgements
Most triton and C++ kernels where taken from https://github.com/Terapines/AI-Benchmark/

The cuTile swiglu implementation is from https://github.com/aghilann/bastile/
