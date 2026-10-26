#ifndef CUDA_FAST_SCORING_HPP
#define CUDA_FAST_SCORING_HPP

#include "cuda_grid.hpp"
#include "cuda_triplet_types.hpp"
#include "interaction_scoring.hpp"
#include "matching_areas.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>


struct GpuFastMatchingArea
{
    GpuPoint3 point{};

    std::uint16_t interaction_bit{};
    std::uint16_t padding[3]{};

    double weight{};
};


struct GpuFastMatchingAreaArrayDevice
{
    GpuFastMatchingArea* areas{nullptr};
    std::uint32_t count{};
};


struct GpuFastScore
{
    std::uint32_t total_areas{};
    std::uint32_t areas_in_grid{};
    std::uint32_t areas_on_occupied_voxel{};
    std::uint32_t compatible_areas{};

    double weighted_score{};

    std::uint32_t valid{};
    std::uint32_t padding{};
};


static_assert(
    std::is_trivially_copyable_v<
        GpuFastMatchingArea
    >
);

static_assert(
    sizeof(GpuFastMatchingArea) == 40
);


static_assert(
    std::is_trivially_copyable_v<
        GpuFastMatchingAreaArrayDevice
    >
);


static_assert(
    std::is_trivially_copyable_v<
        GpuFastScore
    >
);

static_assert(
    sizeof(GpuFastScore) == 32
);


// Pack each logical area in input order with its interaction bit and current
// weight. The kernel adds that weight once for a compatible free voxel, without
// normalization or merging areas that share a physical source area.
std::vector<GpuFastMatchingArea>
make_gpu_fast_matching_areas(
    const std::vector<MatchingArea>& areas,
    const FastScoreWeights& weights
);


#endif