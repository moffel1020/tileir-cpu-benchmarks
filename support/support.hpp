#pragma once
#include <array>
#include <cstdint>
#include <iostream>

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
