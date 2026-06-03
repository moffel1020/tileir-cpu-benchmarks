import cuda.tile as ct
import cuda.tile.compilation as ctc
from utils import make_array_constraint

@ct.kernel
def warp_kernel(
    src_arr,
    offset_arr,
    out_arr,
    channel: ct.Constant[int],
    height: ct.Constant[int],
    width: ct.Constant[int],
):
    pid_h = ct.bid(0)
    pid_c = ct.bid(1)

    h_idx = pid_h

    w_idx = ct.arange(width, dtype=ct.int32)

    offset_val = ct.gather(offset_arr, (h_idx, w_idx)).astype(ct.int16)

    offset_int = (offset_val >> 8).astype(ct.int8)
    offset_fraction = ((offset_val << 8) >> 8).astype(ct.int8)

    indvar = w_idx.astype(ct.int8)

    right_idx = (indvar - offset_int).astype(ct.int8)
    left_idx = (right_idx - 1).astype(ct.int8)

    right_src_w = ct.bitcast(right_idx, ct.uint8).astype(ct.int32)
    left_src_w = ct.bitcast(left_idx, ct.uint8).astype(ct.int32)

    right_val = ct.gather(src_arr, (pid_c, h_idx, right_src_w)).astype(ct.int8)
    left_val = ct.gather(src_arr, (pid_c, h_idx, left_src_w)).astype(ct.int8)

    right_val = ct.where(right_idx < 0, 0, right_val)
    left_val = ct.where(left_idx < 0, 0, left_val)

    out = right_val.astype(ct.int16) << 8
    out += (left_val.astype(ct.int16) - right_val.astype(ct.int16)) * offset_fraction.astype(ct.int16)
    out = (out >> 8).astype(ct.int8)

    ct.scatter(out_arr, (pid_c, h_idx, w_idx), out, check_bounds=False)

def export_warp(file_path: str):
    C, H, W = 3, 512, 512

    ctc.export_kernel(
        kernel=warp_kernel,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(ct.int8, 3, [H * W, W, 1]),
                make_array_constraint(ct.int16, 2),
                make_array_constraint(ct.int8, 3, [H * W, W, 1]),
                ctc.ConstantConstraint(C),
                ctc.ConstantConstraint(H),
                ctc.ConstantConstraint(W),
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="warp",
        )],
        output_file=file_path,
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )
