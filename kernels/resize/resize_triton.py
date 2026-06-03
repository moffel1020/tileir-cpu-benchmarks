import torch
import triton
import triton.language as tl

USE_GPU = False
triton.runtime.driver.set_active_to_cpu()

@triton.jit
def resize_kernel(
    src_ptr,
    out_ptr,
    channel,
    height,
    width,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_h = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    dst_height = 2 * height  # 2x upsample
    dst_width = 2 * width

    hw_fl = 7

    h_idx = pid_h
    input_y = h_idx << (hw_fl - 1)
    y0 = input_y >> hw_fl
    h1_lambda = input_y - (y0 << hw_fl)


    factor = 1 << hw_fl
    h0_lambda = factor - h1_lambda

    y1 = tl.minimum(y0 + 1, height - 1)

    src_offset = pid_c * height * width
    src_ptrs0 = src_ptr + src_offset + y0 * width
    src_ptrs1 = src_ptr + src_offset + y1 * width
    out_ptrs =  out_ptr + (pid_c * dst_height * dst_width + h_idx * dst_width)

    for off in range(0, width * 2, BLOCK_SIZE_W):
        w_idx = off + tl.arange(0, BLOCK_SIZE_W) # [1, BLOCK_SIZE_W]

        mask = (w_idx < dst_width)

        input_x = w_idx << (hw_fl - 1)
        x0 = input_x >> hw_fl
        y0x0 = tl.load(src_ptrs0 + x0, mask=mask, other=0).to(tl.int16)
        y1x0 = tl.load(src_ptrs1 + x0, mask=mask, other=0).to(tl.int16)

        x1 = tl.minimum(x0 + 1, width - 1)
        y0x1 = tl.load(src_ptrs0 + x1, mask=mask, other=0).to(tl.int16)
        y1x1 = tl.load(src_ptrs1 + x1, mask=mask, other=0).to(tl.int16)

        w1_lambda = input_x - (x0 << hw_fl)
        w0_lambda = factor - w1_lambda
        sum1 = (y0x0 * w0_lambda + y0x1 * w1_lambda) >> hw_fl
        sum2 = (y1x0 * w0_lambda + y1x1 * w1_lambda) >> hw_fl
        sum = (sum1 * h0_lambda + sum2 * h1_lambda) >> hw_fl

        sum = sum.to(tl.int8)

        tl.store(out_ptrs + w_idx, sum, mask=mask)


def resize(src_arr, out_arr):
    src_arr = src_arr.contiguous()
    out_arr = out_arr.contiguous()

    # Get dimensions
    channel, height, width = src_arr.shape
    BLOCK_SIZE_W = 64

    # Compute grid dimensions
    grid = lambda meta: (height * 2, channel, 1)

    # Launch the Triton kernel
    kernel = resize_kernel[grid](
        src_arr, out_arr, channel, height, width, BLOCK_SIZE_W
    )
    print(kernel.asm["ttir"])

if __name__ == "__main__":
    C, H, W = 3, 512, 512
    src = torch.ones((C, H, W), dtype=torch.int8, device='cpu')
    out = torch.empty((C, 2 * H, 2 * W), dtype=torch.int8, device='cpu')

    resize(src, out)

def write_resize(file_path: str):
    C, H, W = 3, 512, 512
    BLOCK_SIZE_W = 64

    src_arr = torch.ones((C, H, W), dtype=torch.int8, device='cpu')
    out = torch.empty((C, 2 * H, 2 * W), dtype=torch.int8, device='cpu')
    grid = (H * 2, C, 1)

    kernel = resize_kernel[grid](
        src_arr, out, C, H, W, BLOCK_SIZE_W
    )

    with open(file_path, "w") as f:
        f.write(kernel.asm["ttir"])
