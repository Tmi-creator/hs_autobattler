#pragma once
#include <cstdint>
#include <cassert>

// Оптимизированный PCG32 (Single Sequence)
// Весит всего 8 байт. Идеально ложится в кэш.
struct RngState {
    uint64_t state = 0x853c49e6748fea9bULL;
};

inline uint32_t pcg32_next(RngState& rng) {
    uint64_t oldstate = rng.state;
    // Используем хардкодную константу вместо rng.inc (stream ID)
    rng.state = oldstate * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;

    auto xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    auto rot = static_cast<uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32-rot) & 31));
}

inline void rng_seed(RngState& rng, uint64_t seed) {
    rng.state = seed + 0xda3e39cb94b95bdbULL;
    pcg32_next(rng);
}

// Прямой запрос индекса [0, count-1]. Без лишней математики.
inline int rng_index(RngState& rng, uint32_t count) {
    assert(count > 0 && "rng_index called with 0 or negative count!");
    uint32_t r = pcg32_next(rng);
    // Lemire's trick напрямую для нуля
    return static_cast<int>((static_cast<uint64_t>(r) * count) >> 32);
}

inline int rng_int(RngState& rng, int min_val, int max_val) {
    assert(max_val >= min_val && "rng_int inverted range!");
    auto range = static_cast<uint32_t>(max_val - min_val + 1);
    uint32_t r = pcg32_next(rng);
    uint32_t res = (static_cast<uint64_t>(r) * range) >> 32;
    return min_val + static_cast<int>(res);
}