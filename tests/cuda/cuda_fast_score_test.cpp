#include "cuda_backend.hpp"

#include <cmath>
#include <iostream>
#include <vector>


namespace
{

GpuRigidTransform identity_transform()
{
    GpuRigidTransform transform;


    transform.rotation[0] = 1.0;
    transform.rotation[4] = 1.0;
    transform.rotation[8] = 1.0;

    transform.valid = 1;


    return transform;
}

} // namespace


int main()
{
    GpuProteinGridHost host_grid;


    host_grid.xmin = 0.0;
    host_grid.ymin = 0.0;
    host_grid.zmin = 0.0;

    host_grid.voxel_size = 1.0;

    host_grid.nx = 4;
    host_grid.ny = 4;
    host_grid.nz = 4;


    host_grid.voxels.resize(
        4 * 4 * 4
    );


    const auto index =
        [](
            int x,
            int y,
            int z
        )
        {
            return
                static_cast<std::size_t>(
                    x +
                    4 * (
                        y +
                        4 * z
                    )
                );
        };


    constexpr std::uint16_t bit_a =
        std::uint16_t{1} << 0;

    constexpr std::uint16_t bit_b =
        std::uint16_t{1} << 1;


    /*
     * Voxel (0,0,0):
     * free + compatible with bit_a
     */
    host_grid
        .voxels[
            index(0, 0, 0)
        ]
        .interactions =
            bit_a;


    /*
     * Voxel (1,1,1):
     * occupied.
     *
     * Even if the interaction bit matches,
     * serial semantics say it must not
     * contribute to compatible_areas.
     */
    host_grid
        .voxels[
            index(1, 1, 1)
        ]
        .occupied = 1;

    host_grid
        .voxels[
            index(1, 1, 1)
        ]
        .interactions =
            bit_b;


    GpuProteinGridDevice
        device_grid;


    if (
        !cuda_upload_protein_grid(
            host_grid,
            device_grid
        )
    ) {
        std::cerr
            << "CUDA protein grid upload failed\n";

        return 1;
    }


    const std::vector<GpuFastMatchingArea>
        host_areas = {

            {
                {0.2, 0.2, 0.2},
                bit_a,
                {},
                1.5
            },

            {
                {1.2, 1.2, 1.2},
                bit_b,
                {},
                2.5
            },

            {
                {5.0, 5.0, 5.0},
                bit_a,
                {},
                3.5
            }
        };


    GpuFastMatchingAreaArrayDevice
        device_areas;


    if (
        !cuda_upload_fast_matching_areas(
            host_areas,
            device_areas
        )
    ) {
        std::cerr
            << "CUDA fast-area upload failed\n";

        cuda_free_protein_grid(
            device_grid
        );

        return 1;
    }


    const std::vector<GpuRigidTransform>
        transforms = {

            identity_transform()
        };


    std::vector<GpuFastScore>
        scores;


    if (
        !cuda_score_fast_interactions(
            device_areas,
            device_grid,
            transforms,
            scores
        )
    ) {
        std::cerr
            << "CUDA fast interaction scoring failed\n";

        cuda_free_fast_matching_areas(
            device_areas
        );

        cuda_free_protein_grid(
            device_grid
        );

        return 1;
    }


    if (
        scores.size() != 1
    ) {
        std::cerr
            << "Unexpected CUDA fast score count\n";

        cuda_free_fast_matching_areas(
            device_areas
        );

        cuda_free_protein_grid(
            device_grid
        );

        return 1;
    }


    const auto& score =
        scores[0];


    /*
     * Area 0:
     * inside + free + compatible
     *
     * Area 1:
     * inside + occupied
     *
     * Area 2:
     * outside grid
     */
    if (
        score.valid == 0 ||
        score.total_areas != 3 ||
        score.areas_in_grid != 2 ||
        score.areas_on_occupied_voxel != 1 ||
        score.compatible_areas != 1 ||
        std::abs(
            score.weighted_score -
            1.5
        ) > 1e-12
    ) {
        std::cerr
            << "Unexpected CUDA fast score\n";

        cuda_free_fast_matching_areas(
            device_areas
        );

        cuda_free_protein_grid(
            device_grid
        );

        return 1;
    }


    std::cout
        << "CUDA fast interaction score passed\n";


    cuda_free_fast_matching_areas(
        device_areas
    );

    cuda_free_protein_grid(
        device_grid
    );


    std::cout
        << "CUDA fast scoring tests passed\n";


    return 0;
}