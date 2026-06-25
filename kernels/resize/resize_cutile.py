import cuda.tile as ct
import cuda.tile.compilation as ctc
from utils import make_array_constraint

@ct.kernel
def resize_kernel(
    src_ptr,
    out_ptr,
    channel: ct.Constant[int],
    height: ct.Constant[int],
    width: ct.Constant[int],
    BLOCK_SIZE_W: ct.Constant[int], # TODO: not needed when using tiling
):
    pid_h = ct.bid(0)
    pid_c = ct.bid(1)

    hw_fl = 7

    h_idx = pid_h
    input_y = h_idx << (hw_fl - 1)
    y0 = input_y >> hw_fl
    h1_lambda = input_y - (y0 << hw_fl)

    factor = 1 << hw_fl
    h0_lambda = factor - h1_lambda

    y1 = ct.minimum(y0 + 1, height - 1)

    w_idx = ct.arange(width * 2, dtype=ct.int32)

    input_x = w_idx << (hw_fl - 1)
    x0 = input_x >> hw_fl

    y0x0 = ct.gather(src_ptr, (pid_c, y0, x0), padding_value=0).astype(ct.int16)
    y1x0 = ct.gather(src_ptr, (pid_c, y1, x0), padding_value=0).astype(ct.int16)

    x1 = ct.minimum(x0 + 1, width - 1)

    y0x1 = ct.gather(src_ptr, (pid_c, y0, x1) ,padding_value=0).astype(ct.int16)
    y1x1 = ct.gather(src_ptr, (pid_c, y1, x1) ,padding_value=0).astype(ct.int16)

    w1_lambda = input_x - (x0 << hw_fl)
    w0_lambda = factor - w1_lambda

    sum1 = (y0x0 * w0_lambda + y0x1 * w1_lambda) >> hw_fl
    sum2 = (y1x0 * w0_lambda + y1x1 * w1_lambda) >> hw_fl
    sum_ = (sum1 * h0_lambda + sum2 * h1_lambda) >> hw_fl

    sum_ = sum_.astype(ct.int8)

    ct.scatter(out_ptr, (pid_c, h_idx, w_idx), sum_)

def export_resize(file_path: str):
    C, H, W = 3, 512, 512
    BLOCK_SIZE = 16

    ctc.export_kernel(
        kernel=resize_kernel,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(ct.int8, 3, [H * W, W, 1]),
                make_array_constraint(ct.int8, 3, [H * W * 2 * 2, W * 2, 1]),
                ctc.ConstantConstraint(C),
                ctc.ConstantConstraint(H),
                ctc.ConstantConstraint(W),
                ctc.ConstantConstraint(BLOCK_SIZE),
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="resize",
        )],
        output_file=file_path,
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )
