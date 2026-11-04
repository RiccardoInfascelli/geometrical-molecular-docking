#ifndef GPU_BATCH_RETRY_HPP
#define GPU_BATCH_RETRY_HPP

#include "gpu_ligand_batch.hpp"
#include <vector>

// Only an explicit budget failure permits splitting. Each retry starts after
// the failed call has unwound and released its temporary buffers. Contiguous
// halves preserve record order; a singleton gets one retry with fresh scratch.
template<class Work, class Release>
void run_gpu_batch_adaptive(std::size_t begin, std::size_t end, Work&& work, Release&& release) {
    struct Range { std::size_t first, last; bool single_retry; };
    std::vector<Range> pending;
    if (begin < end) pending.push_back({begin, end, false});
    while (!pending.empty()) {
        const auto range = pending.back();
        pending.pop_back();
        try {
            work(range.first, range.last, range.single_retry);
        } catch (const GpuBatchMemoryError&) {
            if (range.single_retry) throw;
            release();
            if (range.last - range.first == 1) {
                pending.push_back({range.first, range.last, true});
            } else {
                const auto middle = range.first + (range.last - range.first) / 2;
                pending.push_back({middle, range.last, false});
                pending.push_back({range.first, middle, false});
            }
        }
    }
}

#endif
