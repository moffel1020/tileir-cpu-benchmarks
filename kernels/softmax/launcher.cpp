#include <algorithm>
#include <array>
#include <iostream>
#include <omp.h>
#include <random>

#include "../../support/support.hpp"

#define PRINT_STATS
#define LAUNCH_CUTILE

#if defined(LAUNCH_CPP)
void softmax_per_row(float *input, float *out, const int R, const int C);
#include "softmax.hpp"

#elif defined(LAUNCH_TRITON)
extern "C" void softmax_per_row_triton(float *input, float *output,
                                       int32_t pid_x, int32_t pid_y,
                                       int32_t pid_z, uint32_t grid_x,
                                       uint32_t grid_y, uint32_t grid_z);

#elif defined(LAUNCH_CUTILE)
extern "C" void softmax_per_row(float *A, int sizeA1, int sizeA2, int stideA1,
                                int strideA2, float *B, int sizeB1, int sizeB2,
                                int strideB1, int strideB2, uint64_t gridX,
                                uint64_t gridY, uint64_t gridZ);

#endif

constexpr size_t N_REPEAT = 100;

constexpr int N_COLS = 16;
constexpr int N_ROWS = 16;

int main() {
  std::array<float, N_COLS * N_ROWS> input;
  std::array<float, N_COLS * N_ROWS> output;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);
  std::generate(input.begin(), input.end(), [&]() { return dist(gen); });
  // std::iota(input.begin(), input.end(), 0);

  std::fill(output.begin(), output.end(), 0);

#ifdef LAUNCH_CPP
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    softmax_per_row(input.data(), output.data(), N_ROWS, N_COLS);
  });
#endif

#ifdef LAUNCH_TRITON
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr size_t GRID_X = N_ROWS;
    constexpr size_t GRID_Y = 1;
    constexpr size_t GRID_Z = 1;

#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < GRID_Z; z++) {
      for (size_t y = 0; y < GRID_Y; y++) {
        for (size_t x = 0; x < GRID_X; x++) {
          softmax_per_row_triton(input.data(), output.data(), x, y, z, GRID_X,
                                 GRID_Y, GRID_Z);
        }
      }
    }
  });

#endif

#ifdef LAUNCH_CUTILE
  auto times = benchmark<N_REPEAT>([&]() noexcept {
    constexpr size_t GRID_X = N_ROWS;
    constexpr size_t GRID_Y = 1;
    constexpr size_t GRID_Z = 1;

    softmax_per_row(input.data(), N_ROWS, N_COLS, N_COLS, 1, output.data(),
                    N_ROWS, N_COLS, N_COLS, 1, GRID_X, GRID_Y, GRID_Z);
  });
#endif

#ifdef PRINT_STATS
  printStats(times);
#endif

#if 0
  print_array(input);
  std::cout << "\n\n";
  print_array(input);
  std::cout << "\n\n";
#endif
}
