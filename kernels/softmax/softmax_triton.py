import triton
import triton.language as tl
import torch

triton.runtime.driver.set_active_to_cpu() 

@triton.jit
def softmax_per_row_triton(input_ptr, output_ptr, input_row_stride: tl.constexpr, output_row_stride: tl.constexpr, num_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    row_max = -float('inf')
    for off in range(0, n_cols, BLOCK_SIZE):
        col_offsets = off + tl.arange(0, BLOCK_SIZE)
        row = tl.load(row_start_ptr + col_offsets, mask=col_offsets < n_cols, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(row, axis=0))

    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    denominator = 0.0
    for off in range(0, n_cols):
        row = tl.load(row_start_ptr + off)
        # Subtract maximum for numerical stability
        row_minus_max = row - row_max
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator += numerator

        tl.store(output_row_start_ptr + off, numerator)

    for off in range(0, n_cols, BLOCK_SIZE):
        col_offsets = off + tl.arange(0, BLOCK_SIZE)
        row = tl.load(output_row_start_ptr + col_offsets, mask=col_offsets < n_cols, other=-float('inf'))

        softmax_output = row / denominator
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=col_offsets < n_cols)

# @triton.jit
# def softmax_per_row_triton(
#     input_ptr,
#     output_ptr,
#     input_row_stride: tl.constexpr,
#     output_row_stride: tl.constexpr,
#     num_rows,
#     num_cols,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     bidx = tl.program_id(0)
#     num_blocks = tl.num_programs(0)

#     cols = tl.arange(0, BLOCK_SIZE)
#     mask = cols < num_cols

#     for i in range(bidx, num_rows, num_blocks):
#         input_row_ptr = input_ptr + i * input_row_stride
#         output_row_ptr = output_ptr + i * output_row_stride

#         row = tl.load(
#             input_row_ptr + cols,
#             mask=mask,
#             other=-float("inf"),
#         )

#         numerator = tl.exp(row - tl.max(row, axis=0))
#         denominator = tl.sum(numerator, axis=0)

#         tl.store(
#             output_row_ptr + cols,
#             numerator / denominator,
#             mask=mask,
#         )


def softmax(x, y=None):
    if y is None:
        y = torch.empty_like(x)

    num_rows = 512
    num_cols = 2048
    grid = (num_rows,)

    BLOCK_SIZE = triton.next_power_of_2(num_cols)

    kernel = softmax_per_row_triton[grid](
        x,
        y,
        x.stride(0),
        y.stride(0),
        num_rows,
        num_cols,
        BLOCK_SIZE,
    )

    print("-----\nsource\n-----\n", kernel.asm["source"])
    print("-----\nttir\n-----\n", kernel.asm["ttir"])
    print("-----\nttcir\n-----\n", kernel.asm["ttcir"])
    print("-----\ntttcir\n-----\n", kernel.asm["tttcir"])

    return y

def write_softmax_per_row(file_path: str):
    n_rows = 512
    n_cols = 1024

    x = torch.ones((n_rows, n_cols), device='cpu', dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (n_rows,)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # BLOCK_SIZE = 32

    kernel = softmax_per_row_triton[grid](
        x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE
    )


    with open(file_path, "w") as f:
        f.write(kernel.asm["ttir"])


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cpu')
    y_triton_cpu = softmax(x)
