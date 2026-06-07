#include <algorithm>
#include <array>
#include <omp.h>
#include <random>

#include "../../support/support.hpp"

#define LAUNCH_CUTILE
#define PRINT_STATS

#if defined(LAUNCH_CPP)
#include "swiglu.hpp"

#elif defined(LAUNCH_TRITON)
extern "C" void swiglu_fwd_triton(float *gate, float *up, float *out,
                                  int32_t n_cols, uint32_t pid_x,
                                  uint32_t pid_y, uint32_t pid_z,
                                  uint32_t grid_x, uint32_t grid_y,
                                  uint32_t grid_z);

#elif defined(LAUNCH_CUTILE)
extern "C" void swiglu_fwd(float *gate, int sizeG1, int sizeG2, int strideG1,
                           int strideG2, float *up, int sizeU1, int sizeU2,
                           int strideU1, int strideU2, float *out, int sizeO1,
                           int sizeO2, int strideO1, int strideO2,
                           uint64_t gridX, uint64_t gridY, uint64_t gridZ);
#endif

constexpr size_t N_REPEAT = 100;

constexpr size_t H = 512;
constexpr size_t W = 1024;

int main() {
  static std::array<float, H * W> gate;
  static std::array<float, H * W> up;
  static std::array<float, H * W> out;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);

  std::generate(gate.begin(), gate.end(), [&]() { return dist(gen); });
  std::generate(up.begin(), up.end(), [&]() { return dist(gen); });
  std::fill(out.begin(), out.end(), 0);

#ifdef LAUNCH_CPP
  auto times = benchmark<N_REPEAT>(
      [&]() noexcept { swiglu_fwd(gate.data(), up.data(), out.data(), H, W); });
#endif

#ifdef LAUNCH_TRITON
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr size_t GRID_X = H;
    constexpr size_t GRID_Y = 1;
    constexpr size_t GRID_Z = 1;

#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < GRID_Z; z++) {
      for (size_t y = 0; y < GRID_Y; y++) {
        for (size_t x = 0; x < GRID_X; x++) {
          swiglu_fwd_triton(gate.data(), up.data(), out.data(), H, x, y, z,
                            GRID_X, GRID_Y, GRID_Z);
        }
      }
    }
  });
#endif

#ifdef LAUNCH_CUTILE
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr size_t GRID_X = H;
    constexpr size_t GRID_Y = 1;
    constexpr size_t GRID_Z = 1;
    swiglu_fwd(gate.data(), H, W, W, 1, up.data(), H, W, W, 1, out.data(), H, W,
               W, 1, GRID_X, GRID_Y, GRID_Z);
  });
#endif

#ifdef PRINT_STATS
  printStats(times);
#endif

#if 0
  print_array(out, 10);
  std::cout << "\n";
#endif
}
