import triton
import triton.language as tl
import torch

triton.runtime.driver.set_active_to_cpu() 

@triton.jit
def softmax_per_row_triton(
    input_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    num_rows: tl.constexpr, # TODO: these are dynamic in cutile, maybe do the same here?
    num_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bidx = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < num_cols

    for i in range(bidx, num_rows, num_blocks):
        input_row_ptr = input_ptr + i * input_row_stride
        output_row_ptr = output_ptr + i * output_row_stride

        row = tl.load(
            input_row_ptr + cols,
            mask=mask,
            other=-float("inf"),
        )

        numerator = tl.exp(row - tl.max(row, axis=0))
        denominator = tl.sum(numerator, axis=0)

        tl.store(
            output_row_ptr + cols,
            numerator / denominator,
            mask=mask,
        )


def softmax(x, y=None):
    if y is None:
        y = torch.empty_like(x)

    num_rows = 128
    num_cols = 128
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
    n_rows = 16
    n_cols = 16

    x = torch.ones((n_rows, n_cols), device='cpu', dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (n_rows,)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    kernel = softmax_per_row_triton[grid](
        x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE
    )

    print(x)
    print(BLOCK_SIZE)
    print(y)

    with open(file_path, "w") as f:
        f.write(kernel.asm["ttir"])


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cpu')
    y_triton_cpu = softmax(x)
