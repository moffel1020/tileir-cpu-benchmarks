#pragma once

#include <omp.h>

__attribute__((noinline)) void warp(int8_t *src_arr, int16_t *offset_arr,
                                    int8_t *out_arr, size_t channel,
                                    size_t height, size_t width) {
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t c = 0; c < channel; c++) {
    for (size_t h = 0; h < height; h++) {
#pragma omp simd
      for (size_t w = 0; w < width; w++) {
        int16_t offset_val = offset_arr[h * width + w];
        int8_t offset_int = offset_val >> 8;
        int8_t offset_fraction = (offset_val << 8) >> 8;
        int8_t right_idx = w - offset_int;
        int8_t left_idx = right_idx - 1;
        int8_t *src_ptr = &src_arr[c * height * width + h * width];
        int8_t right_val = src_ptr[(uint8_t)right_idx];
        right_val = (right_idx < 0) ? 0 : right_val;
        int8_t left_val = src_ptr[(uint8_t)left_idx];
        left_val = (left_idx < 0) ? 0 : left_val;
        int16_t out = (int16_t)right_val << 8;
        out += (int16_t)(left_val - right_val) * offset_fraction;
        out_arr[c * height * width + h * width + w] = (int8_t)(out >> 8);
      }
    }
  }
}
