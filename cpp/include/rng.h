#pragma once
// rng.h — PCG32 deterministic random number generator
// POD struct, trivially copyable, lives inside GameState
//
// PCG32 = "Permuted Congruential Generator"
// Алгоритм из статьи Мелиссы О'Нил (pcg-random.org).
// По сути: берём линейный конгруэнтный генератор (state * MULT + inc),
// потом "перемешиваем" биты через XOR-shift и rotation.
// Результат: быстрый, качественный рандом в 16 байтах состояния.

#include <cstdint>

struct RngState {
    // Дефолтные значения из спецификации PCG32. Если забудешь вызвать rng_seed(),
    // генератор всё равно выдаст валидную последовательность (а не вырожденную от нулей).
    // При нормальном использовании rng_seed() перезаписывает оба поля.
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc   = 0xda3e39cb94b95bdbULL;
};

// Ядро PCG32 — возвращает случайный uint32_t
// Магические числа из оригинальной статьи — они подобраны для
// максимального периода генератора (2^64) и хороших статистических свойств.
inline uint32_t pcg32_next(RngState& rng) {
    uint64_t oldstate = rng.state;
    // Линейный конгруэнтный шаг: state = state * MULTIPLIER + INCREMENT
    rng.state = oldstate * 6364136223846793005ULL + rng.inc;
    // Перемешивание битов: XOR-shift + rotation
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Инициализация из одного числа-сида
inline void rng_seed(RngState& rng, uint64_t seed) {
    rng.state = 0;
    rng.inc = (seed << 1u) | 1u;
    pcg32_next(rng);
    rng.state += seed;
    pcg32_next(rng);
}

// Случайное целое в диапазоне [min, max] (включительно)
inline int rng_int(RngState& rng, int min_val, int max_val) {
    if (min_val >= max_val) return min_val;
    uint32_t range = static_cast<uint32_t>(max_val - min_val + 1);
    // Rejection sampling — убирает bias при range != степень двойки
    uint32_t threshold = (-range) % range;
    uint32_t r;
    do {
        r = pcg32_next(rng);
    } while (r < threshold);
    return min_val + static_cast<int>(r % range);
}

// Случайный индекс в [0, count-1]
inline int rng_index(RngState& rng, int count) {
    return rng_int(rng, 0, count - 1);
}
