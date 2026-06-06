#pragma once
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

template <size_t N_REPEAT, typename Fn>
inline std::vector<double> benchmark(Fn &&func) {
  std::vector<double> times;
  times.reserve(N_REPEAT);

  // warmup
  for (size_t i = 0; i < N_REPEAT; i++) {
    func();
  }

  for (size_t i = 0; i < N_REPEAT; i++) {
    auto begin = std::chrono::steady_clock::now();
    func();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end - begin;
    times.emplace_back(diff.count());
  }

  return times;
}

inline double getMean(const std::vector<double> &times) {
  auto sum = std::accumulate(times.begin(), times.end(), 0.0);
  return sum / times.size();
}

inline double getStdDev(const std::vector<double> &times) {
  if (times.size() < 2)
    return 0.0;

  double mean = 0.0;
  double m2 = 0.0;
  size_t n = 0;

  for (double x : times) {
    ++n;
    double delta = x - mean;
    mean += delta / n;
    m2 += delta * (x - mean);
  }

  double variance = m2 / (n - 1);
  return std::sqrt(variance);
}

inline void printStats(const std::vector<double> &times) {
  std::cout << getMean(times) << "\n";
  std::cout << getStdDev(times) << "\n";
}

template <typename T, size_t N>
inline void print_array(const std::array<T, N> &arr, int count = N) {
  for (size_t i = 0; i < count; i++) {
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
      std::cout << static_cast<int32_t>(arr[i]) << ' ';
    } else {
      std::cout << arr[i] << ' ';
    }
  }
}
