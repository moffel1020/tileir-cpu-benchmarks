# taken from https://github.com/NVIDIA/cutile-python/blob/main/samples/LayerNorm.py
# commit 83cb18b

import cuda.tile as ct
import cuda.tile.compilation as ctc
from utils import make_array_constraint

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

@ct.kernel
def layernorm_fwd(X, W, B, Y, Mean, Rstd, eps, TILE_N: ConstInt):
    """
    Forward pass: computes mean/var, normalizes input, and applies affine transform.

    Args:
        X: Input tensor (M, N).
        W: Weight tensor (N,).
        B: Bias tensor (N,).
        Y: Output tensor (M, N).
        Mean: Output mean tensor (M,).
        Rstd: Output reciprocal standard deviation tensor (M,).
        eps: Epsilon for numerical stability.
        TILE_N: Tile size along N dimension.
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]

    mean = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute mean
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mean += tx
    mean = ct.sum(mean, axis=1) / N
    ct.store(Mean, index=(bid_m,), tile=mean)

    var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute variance
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mask = (j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)) < N
        centered_tx = ct.where(mask, tx - mean, 0)
        var += centered_tx ** 2
    var = ct.sum(var, axis=1) / N
    rstd = 1 / ct.sqrt(var + eps)
    ct.store(Rstd, index=(bid_m,), tile=rstd)

    for j in range(num_tiles):
        # Normalize and apply affine transformation
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        tb = ct.load(B, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))

def export_layernorm_fwd(file_path: str):
    arr_dtype = ct.float32
    TILE_N = 4096
    M, N = 1024, 4096

    ctc.export_kernel(
        kernel=layernorm_fwd,
        signatures=[ctc.KernelSignature(
            parameters=[
                make_array_constraint(arr_dtype, 2, [N, 1]),
                make_array_constraint(arr_dtype, 1, [1]),
                make_array_constraint(arr_dtype, 1, [1]),
                make_array_constraint(arr_dtype, 2, [N, 1]),
                make_array_constraint(arr_dtype, 1, [1]),
                make_array_constraint(arr_dtype, 1, [1]),
                ctc.ScalarConstraint(dtype=ct.float32),
                ctc.ConstantConstraint(TILE_N),
            ],
            calling_convention=ctc.CallingConvention().cutile_python_v1(),
            symbol="layernorm_fwd",
        )],
        output_file=file_path,
        output_format="tileir_bytecode",
        bytecode_version=None,
        gpu_code="sm_89",
    )
