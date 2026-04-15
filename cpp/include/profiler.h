#pragma once
// profiler.h — минимальный TSC-based профайлер для combat hot path.
// Накапливает такты по именованным секциям в thread_local struct.
// Оверхед на одну секцию ~30 тактов (rdtsc × 2). Мерь только функции от ~500 тактов.
//
// Использование:
//   { ProfScope _(ProfSection::PERFORM_ATTACK); ... }
// В конце прогона вызови g_profiler.dump() / получи через pybind get_profile_data().

#include <cstdint>
#include <x86intrin.h>

enum class ProfSection : int {
    RESOLVE_COMBAT = 0,
    PROCESS_EVENT,
    COLLECT_TRIGGERS,
    SORT_TRIGGERS,
    PERFORM_ATTACK,
    CLEANUP_DEAD,
    CLEANUP_DEAD_WORK,
    FIND_TARGET,
    RECALC_AURAS,
    COUNT
};

struct ProfData {
    uint64_t cycles[(int)ProfSection::COUNT] = {};
    uint64_t calls [(int)ProfSection::COUNT] = {};

    void reset() {
        for (int i = 0; i < (int)ProfSection::COUNT; ++i) {
            cycles[i] = 0;
            calls[i] = 0;
        }
    }
};

extern thread_local ProfData g_prof;

#ifdef PROFILE_COMBAT
struct ProfScope {
    ProfSection sec;
    uint64_t start;
    inline ProfScope(ProfSection s) : sec(s), start(__rdtsc()) {}
    inline ~ProfScope() {
        g_prof.cycles[(int)sec] += __rdtsc() - start;
        g_prof.calls[(int)sec]  += 1;
    }
};
#else
struct ProfScope {
    inline ProfScope(ProfSection) {}
};
#endif
