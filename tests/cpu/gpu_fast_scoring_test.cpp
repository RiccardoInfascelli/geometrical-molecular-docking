#include "cuda_fast_scoring.hpp"

#include "protein_grid.hpp"

#include <cmath>
#include <iostream>
#include <vector>


namespace
{

bool close(
    double a,
    double b
)
{
    return
        std::abs(a - b) <
        1e-12;
}

} // namespace


int main()
{
    const std::vector<MatchingArea>
        areas = {

            {
                0,
                {1.0, 2.0, 3.0},
                InteractionKey::Hydrophobic,
                0.0
            },

            {
                1,
                {-4.0, 5.0, 6.0},
                InteractionKey::HydrogenDonor,
                0.0
            },

            {
                2,
                {7.0, -8.0, 9.0},
                InteractionKey::Aromatic,
                0.0
            }
        };


    FastScoreWeights weights{};


    weights.hydrophobic =
        1.25;

    weights.hydrogen_bond =
        2.50;

    weights.aromatic =
        3.75;


    const auto gpu_areas =
        make_gpu_fast_matching_areas(
            areas,
            weights
        );


    if (
        gpu_areas.size() !=
        areas.size()
    ) {
        std::cerr
            << "Unexpected GPU fast-area count\n";

        return 1;
    }


    // ========================================================
    // Coordinates
    // ========================================================

    if (
        !close(
            gpu_areas[0].point.x,
            1.0
        ) ||
        !close(
            gpu_areas[0].point.y,
            2.0
        ) ||
        !close(
            gpu_areas[0].point.z,
            3.0
        )
    ) {
        std::cerr
            << "GPU fast-area coordinate conversion failed\n";

        return 1;
    }


    // ========================================================
    // Interaction masks
    // ========================================================

    if (
        gpu_areas[0].interaction_bit !=
        static_cast<std::uint16_t>(
            interaction_key_bit(
                InteractionKey::Hydrophobic
            )
        ) ||

        gpu_areas[1].interaction_bit !=
        static_cast<std::uint16_t>(
            interaction_key_bit(
                InteractionKey::HydrogenDonor
            )
        ) ||

        gpu_areas[2].interaction_bit !=
        static_cast<std::uint16_t>(
            interaction_key_bit(
                InteractionKey::Aromatic
            )
        )
    ) {
        std::cerr
            << "GPU fast-area interaction-mask conversion failed\n";

        return 1;
    }


    // ========================================================
    // Weights
    // ========================================================

    if (
        !close(
            gpu_areas[0].weight,
            1.25
        ) ||
        !close(
            gpu_areas[1].weight,
            2.50
        ) ||
        !close(
            gpu_areas[2].weight,
            3.75
        )
    ) {
        std::cerr
            << "GPU fast-area weight conversion failed\n";

        return 1;
    }


    std::cout
        << "GPU fast scoring input conversion passed\n";

    std::cout
        << "areas: "
        << gpu_areas.size()
        << '\n';


    return 0;
}