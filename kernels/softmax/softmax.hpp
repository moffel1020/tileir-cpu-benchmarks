#pragma once

#include <cmath>
#include <omp.h>

__attribute__((noinline)) void softmax_per_row(float *input, float *out, const int R,
                                       const int C) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < R; i++) {
    // find max value in each row
    float *input_r = input + i * C;
    float max_val = input_r[0];
    for (int j = 1; j < C; j++) {
      max_val = std::fmax(input_r[j], max_val);
    }

    // sub maximum and calculate exp
    float sum_exp = 0.0;
    float *out_r = out + i * C;
    for (int j = 0; j < C; j++) {
      out_r[j] = exp(input_r[j] - max_val);
      sum_exp += out_r[j];
    }

    // normalize
    for (int j = 0; j < C; j++) {
      out_r[j] /= sum_exp;
    }
  }
}
