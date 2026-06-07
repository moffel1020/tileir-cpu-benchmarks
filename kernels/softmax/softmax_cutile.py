import cuda.tile as ct
import cuda.tile.compilation as ctc
from utils import make_array_constraint

@ct.kernel
def softmax_1d(input, output, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(input, index=(pid,), shape=(tile_size,))
    numerator = ct.exp(tile)
    denominator = ct.sum(numerator)
    result = numerator / denominator
    ct.store(output, index=(pid,), tile=result)

@ct.kernel
def softmax_per_row(input, output,
                    num_rows: ct.Constant[int],
                    num_cols: ct.Constant[int]):
    bidx = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    for i in range(bidx, num_rows, num_blocks):
        row = ct.load(input, index=(i, 0), shape=(1, num_cols))
        numerator = ct.exp(row - ct.max(row, axis=1, keepdims=True))
        denominator = ct.sum(numerator, axis=1, keepdims=True)
        ct.store(output, index=(i, 0), tile=numerator / denominator)

def export_softmax_1d(file_path: str):
    tile_size = 128
    arr_dtype = ct.float32

    ctc.export_kernel(
        kernel=softmax_1d,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(arr_dtype, 1),
                make_array_constraint(arr_dtype, 1),
                ctc.ConstantConstraint(tile_size),
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="softmax_1d",
        )],
        output_file=file_path, 
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )

def export_softmax_per_row(file_path: str):
    arr_dtype = ct.float32
    n_rows = 512
    n_cols = 1024

    ctc.export_kernel(
        kernel=softmax_per_row,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(arr_dtype, 2, [n_rows, 1]),
                make_array_constraint(arr_dtype, 2, [n_rows, 1]),
                ctc.ConstantConstraint(n_rows),
                ctc.ConstantConstraint(n_cols),
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="softmax_per_row",
        )],
        output_file=file_path, 
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )
