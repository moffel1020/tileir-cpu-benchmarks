#include <algorithm>
#include <array>
#include <omp.h>
#include <random>

#include "../../support/support.hpp"

#define LAUNCH_CUTILE

#if defined(LAUNCH_CPP)
#include "resize.hpp"

#elif defined(LAUNCH_TRITON)
extern "C" void resize_kernel(int8_t *in, int8_t *out, int32_t channel,
                              int32_t height, int32_t width, uint32_t pid_x,
                              uint32_t pid_y, uint32_t pid_z, uint32_t grid_x,
                              uint32_t grid_y, uint32_t grid_z);

#elif defined(LAUNCH_CUTILE)

extern "C" void resize(int8_t *in, uint32_t sizeI1, uint32_t sizeI2,
                       uint32_t sizeI3, uint32_t strideI1, uint32_t strideI2,
                       uint32_t strideI3, int8_t *out, uint32_t sizeO1,
                       uint32_t sizeO2, uint32_t sizeO3, uint32_t strideO1,
                       uint32_t strideO2, uint32_t strideO3, uint64_t gridX,
                       uint64_t gridY, uint64_t gridZ);

#endif

constexpr size_t C = 3;
constexpr size_t H = 512;
constexpr size_t W = 512;

int main() {
  std::array<int8_t, C * H * W> in;
  std::array<int8_t, C *(H * 2) * (W * 2)> out;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int8_t> dist(-128, 127);

  std::generate(in.begin(), in.end(), [&]() { return dist(gen); });
  std::fill(out.begin(), out.end(), 0);

#ifdef LAUNCH_CPP
  resize(in.data(), out.data(), C, H, W);
#endif

#ifdef LAUNCH_TRITON
  constexpr size_t GRID_X = H * 2;
  constexpr size_t GRID_Y = C;
  constexpr size_t GRID_Z = 1;

#pragma omp parallel for collapse(3) schedule(static)
  for (size_t z = 0; z < GRID_Z; z++) {
    for (size_t y = 0; y < GRID_Y; y++) {
      for (size_t x = 0; x < GRID_X; x++) {
        resize_kernel(in.data(), out.data(), C, H, W, x, y, z, GRID_X, GRID_Y,
                      GRID_Z);
      }
    }
  }
#endif

#ifdef LAUNCH_CUTILE
  constexpr size_t GRID_X = H * 2;
  constexpr size_t GRID_Y = C;
  constexpr size_t GRID_Z = 1;
  resize(in.data(), C, H, W, H * W, W, 1, out.data(), C, H * 2, W * 2,
         (H * 2) * (W * 2), W * 2, 1, GRID_X, GRID_Y, GRID_Z);
#endif

#if 1
  print_array(in, 10);
  std::cout << "\n";
  print_array(out);
  std::cout << "\n";
#endif
}
