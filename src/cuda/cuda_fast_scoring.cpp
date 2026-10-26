#include "cuda_fast_scoring.hpp"

#include "protein_grid.hpp"


std::vector<GpuFastMatchingArea>
make_gpu_fast_matching_areas(
    const std::vector<MatchingArea>& areas,
    const FastScoreWeights& weights
)
{
    std::vector<GpuFastMatchingArea> result;

    result.reserve(
        areas.size()
    );


    for (const auto& area :
         areas) {

        GpuFastMatchingArea gpu_area;


        gpu_area.point = {
            static_cast<double>(
                area.centroid.x
            ),
            static_cast<double>(
                area.centroid.y
            ),
            static_cast<double>(
                area.centroid.z
            )
        };


        gpu_area.interaction_bit =
            static_cast<std::uint16_t>(
                interaction_key_bit(
                    area.key
                )
            );


        gpu_area.weight =
            fast_interaction_weight(
                area.key,
                weights
            );


        result.push_back(
            gpu_area
        );
    }


    return result;
}