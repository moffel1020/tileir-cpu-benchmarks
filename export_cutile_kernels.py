from kernels.softmax import softmax_cutile
from kernels.vector_add import vector_add_cutile
from kernels.layernorm import layernorm_cutile
from kernels.swiglu import swiglu_cutile
from kernels.warp import warp_cutile
from kernels.resize import resize_cutile
from kernels.sharpen import sharpen_cutile

import subprocess

CUDA_TILE_CPU = "/home/thomas/repos/cuda-tile-cpu/build/bin"
CUDA_TILE_OPT = CUDA_TILE_CPU + "/cuda-tile-opt"
CUDA_TILE_TR = CUDA_TILE_CPU + "/cuda-tile-translate"

def bytecode_to_mlir(bytecode_file_path: str) -> str:
    output = subprocess.run(
        [CUDA_TILE_TR, "--cudatilebc-to-mlir", bytecode_file_path], 
        capture_output=True, text=True)

    if output.returncode != 0:
        print("Error:", output.stderr)

    return output.stdout

kernels = [
    # {
    #     "export_fn": vector_add_cutile.export_vec_add_1d, 
    #     "bc_path": "kernel_ir/cutile/vector_add_1d.bc", 
    #     "mlir_path": "kernel_ir/cutile/vector_add_1d.mlir"
    # },
    # {
    #     "export_fn": vector_add_cutile.export_vec_add_1d_gather, 
    #     "bc_path": "kernel_ir/cutile/vector_add_1d_gather.bc", 
    #     "mlir_path": "kernel_ir/cutile/vector_add_1d_gather.mlir"
    # },
    {
        "export_fn": softmax_cutile.export_softmax_1d, 
        "bc_path": "kernels_ir/cutile/softmax_1d.bc", 
        "mlir_path": "kernels_ir/cutile/softmax_1d.mlir"
    },
    {
        "export_fn": softmax_cutile.export_softmax_per_row, 
        "bc_path": "kernels_ir/cutile/softmax_per_row.bc", 
        "mlir_path": "kernels_ir/cutile/softmax_per_row.mlir"
    },
    {
        "export_fn": layernorm_cutile.export_layernorm_fwd, 
        "bc_path": "kernels_ir/cutile/layernorm_fwd.bc", 
        "mlir_path": "kernels_ir/cutile/layernorm_fwd.mlir"
    },
    {
        "export_fn": swiglu_cutile.export_swiglu_fwd, 
        "bc_path": "kernels_ir/cutile/swiglu_fwd.bc", 
        "mlir_path": "kernels_ir/cutile/swiglu_fwd.mlir"
    },
    {
        "export_fn": warp_cutile.export_warp,
        "bc_path": "kernels_ir/cutile/warp.bc", 
        "mlir_path": "kernels_ir/cutile/warp.mlir"
    },
    {
        "export_fn": resize_cutile.export_resize,
        "bc_path": "kernels_ir/cutile/resize.bc", 
        "mlir_path": "kernels_ir/cutile/resize.mlir"
    },
    {
        "export_fn": sharpen_cutile.export_sharpen,
        "bc_path": "kernels_ir/cutile/sharpen_3x3.bc", 
        "mlir_path": "kernels_ir/cutile/sharpen_3x3.mlir"
    },
]

if __name__ == "__main__":
    for k in kernels:
        bc_path = k["bc_path"]
        mlir_path = k["mlir_path"]
        export_fn = k["export_fn"]

        export_fn(bc_path)
        mlir_text = bytecode_to_mlir(bc_path)
        with open(mlir_path, "w") as f:
            f.write(mlir_text)
