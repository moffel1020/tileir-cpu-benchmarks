#include <algorithm>
#include <array>
#include <omp.h>
#include <random>

#include "../../support/support.hpp"

#define LAUNCH_CUTILE
#define PRINT_STATS

#if defined(LAUNCH_CPP)
#include "warp.hpp"

#elif defined(LAUNCH_TRITON)
extern "C" void warp_kernel(int8_t *in, int16_t *off, int8_t *out,
                            int32_t channel, int32_t height, int32_t width,
                            uint32_t pid_x, uint32_t pid_y, uint32_t pid_z,
                            uint32_t grid_x, uint32_t grid_y, uint32_t grid_z);

#elif defined(LAUNCH_CUTILE)
extern "C" void warp(int8_t *in, uint32_t sizeI1, uint32_t sizeI2,
                     uint32_t sizeI3, uint32_t strideI1, uint32_t strideI2,
                     uint32_t strideI3, int16_t *off, uint32_t sizeOff1,
                     uint32_t sizeOff2, uint32_t strideOff1,
                     uint32_t strideOff2, int8_t *out, uint32_t sizeO1,
                     uint32_t sizeO2, uint32_t sizeO3, uint32_t strideO1,
                     uint32_t strideO2, uint32_t strideO3, uint64_t gridX,
                     uint64_t gridY, uint64_t gridZ);

#endif

constexpr size_t N_REPEAT = 1000;

constexpr size_t C = 3;
constexpr size_t H = 2048;
constexpr size_t W = 2048;

int main() {
  static std::array<int8_t, C * H * W> in;
  static std::array<int16_t, H * W> off;
  static std::array<int8_t, C * H * W> out;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int8_t> dist(-128, 127);
  std::generate(in.begin(), in.end(), [&]() { return dist(gen); });

  std::uniform_int_distribution<int8_t> distInt(-128, 127);
  std::uniform_int_distribution<int8_t> distFrac(0, 127);
  std::generate(off.begin(), off.end(), [&]() {
    return (static_cast<int32_t>(distInt(gen)) << 8) | distFrac(gen);
  });

  std::fill(out.begin(), out.end(), 0);

#ifdef LAUNCH_CPP
  auto times = benchmark<N_REPEAT>(
      [&]() noexcept { warp(in.data(), off.data(), out.data(), C, H, W); });
#endif

#ifdef LAUNCH_TRITON
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr size_t GRID_X = H;
    constexpr size_t GRID_Y = C;
    constexpr size_t GRID_Z = 1;

#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < GRID_Z; z++) {
      for (size_t y = 0; y < GRID_Y; y++) {
        for (size_t x = 0; x < GRID_X; x++) {
          warp_kernel(in.data(), off.data(), out.data(), C, H, W, x, y, z,
                      GRID_X, GRID_Y, GRID_Z);
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

    warp(in.data(), C, H, W, H * W, W, 1, off.data(), H, W, W, 1, out.data(), C,
         H, W, H * W, W, 1, GRID_X, GRID_Y, GRID_Z);
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
