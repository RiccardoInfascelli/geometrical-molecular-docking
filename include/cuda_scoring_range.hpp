#ifndef CUDA_SCORING_RANGE_HPP
#define CUDA_SCORING_RANGE_HPP

#include <cstdint>
#include <type_traits>

// One slice per selected score row. Chemical features belong to [begin,
// begin + count); pose_begin offsets that row's local survivor ID.
struct GpuLigandScoringRange {
    std::uint32_t begin{}, count{}, pose_begin{};
};

static_assert(std::is_trivially_copyable_v<GpuLigandScoringRange>);
static_assert(sizeof(GpuLigandScoringRange) == 12);

#endif
