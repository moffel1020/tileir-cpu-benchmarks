import torch
import triton
import triton.language as tl
import os

USE_GPU = False
triton.runtime.driver.set_active_to_cpu()

@triton.jit
def warp_kernel(
    src_ptr,        # *int8, shape [C, H, W]
    offset_ptr,     # *int16, shape [H, W]
    out_ptr,        # *int8, shape [C, H, W]
    channel,        # int
    height,         # int
    width,          # int
    BLOCK_SIZE_W: tl.constexpr,
):

    pid_h = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    # Compute the indices
    h_idx = pid_h

    for off in range(0, width, BLOCK_SIZE_W):
        w_idx = off + tl.arange(0, BLOCK_SIZE_W)
        # Create a mask to avoid out-of-bounds accesses
        mask = (w_idx < width)

        # Compute offset indices
        offset_idx = h_idx * width + w_idx  # [BLOCK_SIZE_H, BLOCK_SIZE_W]

        # Load offset values
        offset_val = tl.load(offset_ptr + offset_idx, mask=mask, other=0).to(tl.int16)

        # Decompose offset_val into integer and fractional parts
        offset_int = (offset_val >> 8).to(tl.int8)
        offset_fraction = ((offset_val << 8) >> 8).to(tl.int8)

        # Compute indvar (w_idx)
        indvar = w_idx.to(tl.int8)

        # Compute right_idx and left_idx
        right_idx = (indvar - offset_int).to(tl.int8)
        left_idx = (right_idx - 1).to(tl.int8)

        # Compute src indices
        src_base = pid_c * height * width + h_idx * width  # [BLOCK_SIZE_H, 1]
        right_src_idx = src_base + right_idx  # [BLOCK_SIZE_H, BLOCK_SIZE_W]
        left_src_idx = src_base + left_idx

        # Load values
        right_val = tl.load(src_ptr + right_src_idx, mask=mask, other=0).to(tl.int8)
        left_val = tl.load(src_ptr + left_src_idx, mask=mask, other=0).to(tl.int8)

        right_val = tl.where(right_idx < 0, 0, right_val)
        left_val = tl.where(left_idx < 0, 0, left_val)

        # Compute output
        out = (right_val.to(tl.int16) << 8)
        out += (left_val - right_val).to(tl.int16) * offset_fraction.to(tl.int16)
        out = (out >> 8).to(tl.int8)

        # Compute output indices
        out_idx = pid_c * height * width + h_idx * width + w_idx

        # Store the result
        tl.store(out_ptr + out_idx, out, mask=mask)

def warp(src_arr, offset_arr, out_arr):
    src_arr = src_arr.contiguous()
    offset_arr = offset_arr.contiguous()
    out_arr = out_arr.contiguous()

    # Get dimensions
    channel, height, width = src_arr.shape

    # Compute grid dimensions
    grid = lambda meta: (height, channel, 1)

    BLOCK_SIZE = 128
    # Launch the Triton kernel
    kernel = warp_kernel[grid](
        src_arr, offset_arr, out_arr, channel, height, width, BLOCK_SIZE
    )
    print(kernel.asm["ttir"])

def write_warp(file_path: str):
    C, H, W = 3, 2048, 2048
    BLOCK_SIZE = 16

    src = torch.ones((C, H, W), dtype=torch.int8, device='cpu')
    offset = torch.zeros((H, W), dtype=torch.int16, device='cpu')  # Example offset values
    out = torch.empty((C, H, W), dtype=torch.int8, device='cpu')

    grid = (H, C, 1)

    kernel = warp_kernel[grid](
        src, offset, out, C, H, W, BLOCK_SIZE
    )

    with open(file_path, "w") as f:
        f.write(kernel.asm["ttir"])


if __name__ == "__main__":
    C, H, W = 3, 512, 512
    src = torch.ones((C, H, W), dtype=torch.int8, device='cpu')
    offset = torch.zeros((H, W), dtype=torch.int16, device='cpu')  # Example offset values
    out = torch.empty((C, H, W), dtype=torch.int8, device='cpu')
    warp(src, offset, out)
