#pragma once

#include <algorithm>
#include <omp.h>

static constexpr int16_t clamp_i16(int16_t x, int16_t lo, int16_t hi) {
  return std::max(lo, std::min(x, hi));
}

__attribute__((noinline)) void sharpen_3x3(uint8_t *src_arr, uint8_t *out_arr,
                                          size_t channel, size_t height,
                                          size_t width) {
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t c = 0; c < channel; c++) {
    for (size_t h = 0; h < height; h++) {
      size_t hm1 = (h == 0) ? 0 : h - 1;
      size_t hp1 = std::min(h + 1, height - 1);

      uint8_t *src_base = src_arr + c * height * width;
      uint8_t *out_base = out_arr + c * height * width;

#pragma omp simd
      for (size_t w = 0; w < width; w++) {
        size_t wm1 = (w == 0) ? 0 : w - 1;
        size_t wp1 = std::min(w + 1, width - 1);

        int16_t center = src_base[h * width + w];
        int16_t up = src_base[hm1 * width + w];
        int16_t down = src_base[hp1 * width + w];
        int16_t left = src_base[h * width + wm1];
        int16_t right = src_base[h * width + wp1];

        int16_t val = 5 * center - up - down - left - right;
        val = clamp_i16(val, 0, 255);

        out_base[h * width + w] = static_cast<uint8_t>(val);
      }
    }
  }
}
