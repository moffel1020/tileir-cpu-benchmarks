import cuda.tile as ct
import cuda.tile.compilation as ctc
from utils import make_array_constraint

@ct.kernel
def sharpen_3x3(src, out, W: ct.Constant[int]):
    h = ct.bid(0)
    c = ct.bid(1)

    # check_bounds defaults to true, so no explicit mask is needed like in triton
    offsets = ct.arange(W, dtype=ct.int32)

    center = ct.gather(
        src,
        (c, h, offsets),
    ).astype(ct.int16)

    up = ct.gather(
        src,
        (c, h - 1, offsets),
    ).astype(ct.int16)

    down = ct.gather(
        src,
        (c, h + 1, offsets),
    ).astype(ct.int16)

    left = ct.gather(
        src,
        (c, h, offsets-1),
    ).astype(ct.int16)

    right = ct.gather(
        src,
        (c, h, offsets+1),
    ).astype(ct.int16)

    # 0 -1  0
    # -1 5 -1
    # 0 -1  0
    val = 5 * center - up - down - left - right

    val = ct.minimum(ct.maximum(val, 0), 255)
    val = val.astype(ct.uint8)

    ct.scatter(
        out,
        (c, h, offsets),
        val,
    )

def export_sharpen(file_path: str):
    C, H, W = 3, 512, 512

    ctc.export_kernel(
        kernel=sharpen_3x3,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(ct.uint8, 3, [H * W, W, 1]),
                make_array_constraint(ct.uint8, 3, [H * W, W, 1]),
                ctc.ConstantConstraint(W),
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="sharpen_3x3",
        )],
        output_file=file_path,
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )
