import cuda.tile as ct
import cuda.tile.compilation as ctc
from utils import make_array_constraint


@ct.kernel
def vector_add_1d(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

@ct.kernel
def vector_add_1d_gather(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    indices = pid * tile_size + ct.arange(tile_size, dtype=ct.int32)

    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)

    result = a_tile + b_tile
    ct.scatter(c, indices, result)


def export_vector_add_1d(file_path: str):
    tile_size = 128
    arr_dtype = ct.float32

    ctc.export_kernel(
        kernel=vector_add_1d,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(arr_dtype, 1),
                make_array_constraint(arr_dtype, 1),
                make_array_constraint(arr_dtype, 1),
                ctc.ConstantConstraint(tile_size)
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="vector_add_1d",
        )],
        output_file=file_path,
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )

def export_vector_add_1d_gather(file_path: str):
    tile_size = 128
    arr_dtype = ct.float32

    ctc.export_kernel(
        kernel=vector_add_1d_gather,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(arr_dtype, 1),
                make_array_constraint(arr_dtype, 1),
                make_array_constraint(arr_dtype, 1),
                ctc.ConstantConstraint(tile_size)
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="vector_add_1d_gather",
        )],
        output_file=file_path,
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )
