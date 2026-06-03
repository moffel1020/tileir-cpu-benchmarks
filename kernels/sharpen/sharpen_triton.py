import torch
import triton
import triton.language as tl

USE_GPU = False
triton.runtime.driver.set_active_to_cpu()

@triton.jit
def sharpen_3x3_kernel(
    src,
    out,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_c = tl.program_id(1)

    w = tl.program_id(2) * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < W

    h = pid_h

    hm1 = tl.maximum(h - 1, 0)
    hp1 = tl.minimum(h + 1, H - 1)

    wm1 = tl.maximum(w - 1, 0)
    wp1 = tl.minimum(w + 1, W - 1)

    channel = pid_c * H * W

    center = tl.load(src + channel + h * W + w, mask=mask, other=0).to(tl.int16)
    up     = tl.load(src + channel + hm1 * W + w, mask=mask, other=0).to(tl.int16)
    down   = tl.load(src + channel + hp1 * W + w, mask=mask, other=0).to(tl.int16)
    left   = tl.load(src + channel + h * W + wm1, mask=mask, other=0).to(tl.int16)
    right  = tl.load(src + channel + h * W + wp1, mask=mask, other=0).to(tl.int16)

    # 3x3 sharpen kernel:
    #
    #   0  -1   0
    #  -1   5  -1
    #   0  -1   0
    #
    val = 5 * center - up - down - left - right

    # For int8 images.
    val = tl.minimum(tl.maximum(val, -128), 127).to(tl.int8)

    tl.store(out + channel + h * W + w, val, mask=mask)


def sharpen3x3(src: torch.Tensor, block_w: int = 128):
    assert src.ndim == 3
    assert src.dtype == torch.int8
    assert src.is_contiguous()

    C, H, W = src.shape
    out = torch.empty_like(src)

    grid = (
        H,
        C,
        triton.cdiv(W, block_w),
    )

    kernel = sharpen_3x3_kernel[grid](
        src,
        out,
        C, H, W,
        BLOCK_W=block_w,
    )

    print(kernel.asm["ttir"])

    return out

def write_sharpen_3x3(file_path: str):
    C, H, W = 3, 512, 512
    BLOCK_W = 512

    src = torch.ones((C, H, W), device="cpu", dtype=torch.int8)
    out = torch.empty_like(src)

    grid = (H, C, triton.cdiv(W, BLOCK_W))
    kernel = sharpen_3x3_kernel[grid](
        src, out, C, H, W, BLOCK_W
    )
    
    with open(file_path, "w") as f:
        f.write(kernel.asm["ttir"])


if __name__ == "__main__":
    C, H, W = 3, 512, 512
    src = torch.ones((C, H, W), device="cpu", dtype=torch.int8)
    out = sharpen3x3(src)
