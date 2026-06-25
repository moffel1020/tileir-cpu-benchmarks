import triton
import triton.language as tl
import torch


@triton.jit
def swiglu_fwd_triton(
    gate_ptr,
    up_ptr,
    output_ptr,
    n_cols,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    bid = tl.program_id(0)

    for i in range(0, width, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        gate_tile = tl.load(
            gate_ptr + bid * width + offsets,
            mask=mask,
            other=0.0,
        )

        up_tile = tl.load(
            up_ptr + bid * width + offsets,
            mask=mask,
            other=0.0,
        )

        # Compute sigmoid in float32 for numerical stability.
        gate_f32 = gate_tile.to(tl.float32)

        denom = 1.0 + tl.exp(-gate_f32)
        sig = 1.0 / denom

        # SiLU(gate) * up
        silu_gate = gate_f32 * sig

        # Cast back to gate dtype before multiplying, matching:
        result = silu_gate.to(gate_tile.dtype) * up_tile

        tl.store(
            output_ptr + bid * width + offsets,
            result,
            mask=mask,
        )

def test_swiglu_fwd():
    gate = torch.randn((1024, 4096), device="cpu", dtype=torch.float32)
    up = torch.randn((1024, 4096), device="cpu", dtype=torch.float32)
    output = torch.empty_like(gate)

    TILE_SIZE = 1024
    BLOCK_SIZE = 16
    grid = (4096,)

    n_rows, n_cols = gate.shape

    kernel = swiglu_fwd_triton[(grid,)](
        gate,
        up,
        output,
        n_cols,
        TILE_SIZE,
        BLOCK_SIZE,
    )

    print(kernel.asm["ttir"])
    return output

def write_swiglu(file_path: str):
    gate = torch.randn((512, 1024), device="cpu", dtype=torch.float32)
    up = torch.randn((512, 1024), device="cpu", dtype=torch.float32)
    output = torch.empty_like(gate)

    TILE_SIZE = 1024
    BLOCK_SIZE = 16
    grid = (512,)

    _, n_cols = gate.shape

    kernel = swiglu_fwd_triton[grid](
        gate,
        up,
        output,
        n_cols,
        TILE_SIZE,
        BLOCK_SIZE,
    )

    with open(file_path, "w") as f:
        f.write(kernel.asm["ttir"])


if __name__ == "__main__":
    test_swiglu_fwd()
