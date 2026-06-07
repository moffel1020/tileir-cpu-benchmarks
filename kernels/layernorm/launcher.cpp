#include <algorithm>
#include <array>
#include <omp.h>
#include <random>

#include "../../support/support.hpp"

#define PRINT_STATS
// #define LAUNCH_CUTILE

#if defined(LAUNCH_CPP)
#include "layernorm.hpp"

#elif defined(LAUNCH_TRITON)
extern "C" void _layer_norm_fwd_fused(float *X, float *Y, float *W, float *B,
                                      float *mean, float *rstd, uint32_t stride,
                                      uint32_t N, uint32_t pid_x,
                                      uint32_t pid_y, uint32_t pid_z,
                                      uint32_t grid_x, uint32_t grid_y,
                                      uint32_t grid_z);

#elif defined(LAUNCH_CUTILE)
extern "C" void layernorm_fwd(float *X, int sizeX1, int sizeX2, int strideX1,
                              int strideX2, float *W, int sizeW1, int strideW1,
                              float *B, int sizeB1, int strideB1, float *Y,
                              int sizeY1, int sizeY2, int strideY1,
                              int strideY2, float *mean, int sizeM1,
                              int strideM1, float *rstd, int sizeR1,
                              int strideR1, uint64_t gridX, uint64_t gridY,
                              uint64_t gridZ);
#endif

constexpr size_t N_REPEAT = 1000;

constexpr size_t M = 1024;
constexpr size_t N = 4096;

int main() {
  static std::array<float, M * N> X;
  static std::array<float, N> W;
  static std::array<float, N> B;
  static std::array<float, M * N> Y;
  static std::array<float, M> mean;
  static std::array<float, M> rstd;

  std::fill(X.begin(), X.end(), 1);
  std::fill(W.begin(), W.end(), 1);
  std::fill(B.begin(), B.end(), 1);

  std::fill(Y.begin(), Y.end(), 0);
  std::fill(mean.begin(), mean.end(), 0);
  std::fill(rstd.begin(), rstd.end(), 0);

#ifdef LAUNCH_CPP
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    layernorm_forward(Y.data(), mean.data(), rstd.data(), X.data(), W.data(),
                      B.data(), M, N);
  });
#endif

#ifdef LAUNCH_TRITON
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr uint32_t GRID_X = M;
    constexpr uint32_t GRID_Y = 1;
    constexpr uint32_t GRID_Z = 1;

#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < GRID_Z; z++) {
      for (size_t y = 0; y < GRID_Y; y++) {
        for (size_t x = 0; x < GRID_X; x++) {
          _layer_norm_fwd_fused(X.data(), Y.data(), W.data(), B.data(),
                                mean.data(), rstd.data(), 1, N, x, y, z, GRID_X,
                                GRID_Y, GRID_Z);
        }
      }
    }
  });
#endif

#ifdef LAUNCH_CUTILE
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr uint64_t GRID_X = M;
    constexpr uint64_t GRID_Y = 1;
    constexpr uint64_t GRID_Z = 1;

    layernorm_fwd(X.data(), M, N, N, 1, W.data(), N, 1, B.data(), N, 1,
                  Y.data(), M, N, N, 1, mean.data(), M, 1, rstd.data(), M, 1,
                  GRID_X, GRID_Y, GRID_Z);
  });
#endif

#ifdef PRINT_STATS
  printStats(times);
#endif

#if 0
  print_array(Y, 100);
  std::cout << "\n\n";
  print_array(mean, 100);
  std::cout << "\n\n";
  print_array(rstd, 100);
  std::cout << "\n\n";
#endif
}
