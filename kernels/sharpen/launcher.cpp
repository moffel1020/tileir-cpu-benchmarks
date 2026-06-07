#include <algorithm>
#include <array>
#include <omp.h>
#include <random>

#include "../../support/support.hpp"

#define PRINT_STATS
#define LAUNCH_CPP

#if defined(LAUNCH_CPP)
#include "sharpen.hpp"

#elif defined(LAUNCH_TRITON)
extern "C" void sharpen_3x3_kernel(uint8_t *in, uint8_t *out, uint32_t pid_x,
                                   uint32_t pid_y, uint32_t pid_z,
                                   uint32_t grid_x, uint32_t grid_y,
                                   uint32_t grid_z);

#elif defined(LAUNCH_CUTILE)
extern "C" void sharpen_3x3(uint8_t *in, uint32_t sizeI1, uint32_t sizeI2,
                            uint32_t sizeI3, uint32_t strideI1,
                            uint32_t strideI2, uint32_t strideI3, uint8_t *out,
                            uint32_t sizeO1, uint32_t sizeO2, uint32_t sizeO3,
                            uint32_t strideO1, uint32_t strideO2,
                            uint32_t strideO3, uint64_t gridX, uint64_t gridY,
                            uint64_t gridZ);

#endif

constexpr size_t N_REPEAT = 1000;

constexpr size_t C = 3;
constexpr size_t H = 2048;
constexpr size_t W = 2048;

int main() {
  // TODO: probs broken kernels

  static std::array<uint8_t, C * H * W> in;
  static std::array<uint8_t, C * H * W> out;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dist(1, 255);

  std::generate(in.begin(), in.end(), [&]() { return dist(gen); });
  std::fill(out.begin(), out.end(), 0);

#ifdef LAUNCH_CPP
  auto times = benchmark<N_REPEAT>(
      [&]() noexcept { sharpen_3x3(in.data(), out.data(), C, H, W); });
#endif

#ifdef LAUNCH_TRITON
  constexpr size_t BLOCK_W = 16;

  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr size_t GRID_X = H;
    constexpr size_t GRID_Y = C;
    constexpr size_t GRID_Z = W / BLOCK_W;

#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < GRID_Z; z++) {
      for (size_t y = 0; y < GRID_Y; y++) {
        for (size_t x = 0; x < GRID_X; x++) {
          sharpen_3x3_kernel(in.data(), out.data(), x, y, z, GRID_X, GRID_Y,
                             GRID_Z);
        }
      }
    }
  });
#endif

#ifdef LAUNCH_CUTILE
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr size_t GRID_X = H;
    constexpr size_t GRID_Y = C;
    constexpr size_t GRID_Z = 1;
    sharpen_3x3(in.data(), C, H, W, H * W, W, 1, out.data(), C, H, W, H * W, W,
                1, GRID_X, GRID_Y, GRID_Z);
  });
#endif

#ifdef PRINT_STATS
  printStats(times);
#endif

#if 0
  print_array(in, 100);
  std::cout << "\n\n";
  print_array(out, 100);
  std::cout << "\n";
#endif
}