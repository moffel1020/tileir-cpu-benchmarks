import cuda.tile as ct
import cuda.tile.compilation as ctc
from utils import make_array_constraint

ConstInt = ct.Constant[int]

@ct.kernel
def swiglu_fwd(
    gate,
    up,
    output,
    WIDTH: ConstInt,
):
    bid = ct.bid(0)

    gate_tile = ct.load(gate, (bid, 0), (1, WIDTH))
    up_tile = ct.load(up, (bid, 0), (1, WIDTH))

    # Compute sigmoid in float32 for numerical stability
    gate_f32 = ct.astype(gate_tile, ct.float32)
    denom = ct.add(1.0, ct.exp(-gate_f32), flush_to_zero=True)
    sig = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)

    # SiLU(gate) * up
    silu_gate = ct.mul(gate_f32, sig, flush_to_zero=True)
    result = ct.astype(silu_gate, gate.dtype) * up_tile

    ct.store(output, (bid, 0), result)

def export_swiglu_fwd(file_path: str):
    arr_dtype = ct.float32
    H, W = 512, 1024

    ctc.export_kernel(
        kernel=swiglu_fwd,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(arr_dtype, 2, [W, 1]),
                make_array_constraint(arr_dtype, 2, [W, 1]),
                make_array_constraint(arr_dtype, 2, [W, 1]),
                ctc.ConstantConstraint(W)
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="swiglu_fwd",
        )],
        output_file=file_path,
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )