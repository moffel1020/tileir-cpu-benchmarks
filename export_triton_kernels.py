from kernels.resize import resize_triton
from kernels.sharpen import sharpen_triton
from kernels.softmax import softmax_triton
from kernels.layernorm import layernorm_triton
from kernels.swiglu import swiglu_triton
from kernels.warp import warp_triton

kernels = [
    {
        "write_fn": resize_triton.write_resize,
        "mlir_path": "kernels_ir/triton/resize.mlir"
    },
    {
        "write_fn": sharpen_triton.write_sharpen_3x3,
        "mlir_path": "kernels_ir/triton/sharpen_3x3.mlir"
    },

    {
        "write_fn": softmax_triton.write_softmax_per_row,
        "mlir_path": "kernels_ir/triton/softmax_per_row.mlir"
    },
    {
        "write_fn": layernorm_triton.write_layernorm_fwd,
        "mlir_path": "kernels_ir/triton/layernorm_fwd.mlir"
    },
    {
        "write_fn": swiglu_triton.write_swiglu,
        "mlir_path": "kernels_ir/triton/swiglu.mlir"
    },
    {
        "write_fn": warp_triton.write_warp,
        "mlir_path": "kernels_ir/triton/warp.mlir"
    },
]

if __name__ == "__main__":
    for k in kernels:
        write_fn = k["write_fn"]
        mlir_path = k["mlir_path"]
        write_fn(mlir_path)
