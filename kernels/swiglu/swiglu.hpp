#include <cmath>
#include <omp.h>

__attribute__((noinline)) void swiglu_fwd(float *gate_arr, float *up_arr,
                                          float *out_arr, size_t n_rows,
                                          size_t n_cols) {
#pragma omp parallel for schedule(static)
  for (size_t bid = 0; bid < n_rows; bid++) {
    float *gate_ptr = gate_arr + bid * n_cols;
    float *up_ptr = up_arr + bid * n_cols;
    float *out_ptr = out_arr + bid * n_cols;

#pragma omp simd
    for (size_t col = 0; col < n_cols; col++) {
      float gate = gate_ptr[col];
      float up = up_ptr[col];

      float sig = 1.0f / (1.0f + std::exp(-gate));
      float silu_gate = gate * sig;

      out_ptr[col] = silu_gate * up;
    }
  }
}
