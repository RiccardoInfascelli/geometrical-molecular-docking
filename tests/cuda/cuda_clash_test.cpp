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
    // ========================================================
    // Synthetic protein grid
    // ========================================================

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


    /*
     * Occupied voxels:
     *
     * (1,1,1)
     * (2,1,1)
     */
    host_grid
        .voxels[
            index(1, 1, 1)
        ]
        .occupied = 1;

    host_grid
        .voxels[
            index(2, 1, 1)
        ]
        .occupied = 1;


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


    // ========================================================
    // Ligand atoms
    // ========================================================

    const std::vector<GpuPoint3>
        ligand_atoms = {

            {1.2, 1.2, 1.2},
            {0.2, 0.2, 0.2},
            {5.0, 5.0, 5.0}
        };


    GpuPointArrayDevice
        device_ligand_atoms;


    if (
        !cuda_upload_points(
            ligand_atoms,
            device_ligand_atoms
        )
    ) {
        std::cerr
            << "CUDA ligand atom upload failed\n";

        cuda_free_protein_grid(
            device_grid
        );

        return 1;
    }


    // ========================================================
    // Pose 0:
    //
    // identity
    //
    // atom 0 -> occupied
    // atom 1 -> free
    // atom 2 -> outside grid
    //
    // occupied = 1 / 3
    // ========================================================

    auto transform0 =
        identity_transform();


    // ========================================================
    // Pose 1:
    //
    // translate +1 along X
    //
    // atom 0:
    // (1.2,1.2,1.2)
    // -> (2.2,1.2,1.2)
    // -> occupied
    //
    // atom 1:
    // (0.2,0.2,0.2)
    // -> (1.2,0.2,0.2)
    // -> free
    //
    // atom 2 remains outside
    //
    // occupied = 1 / 3
    // ========================================================

    auto transform1 =
        identity_transform();

    transform1.translation[0] =
        1.0;


    const std::vector<GpuRigidTransform>
        transforms = {

            transform0,
            transform1
        };


    std::vector<GpuClashScore>
        scores;


    if (
        !cuda_score_grid_clash(
            device_ligand_atoms,
            device_grid,
            transforms,
            scores
        )
    ) {
        std::cerr
            << "CUDA grid clash failed\n";

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_protein_grid(
            device_grid
        );

        return 1;
    }


    if (
        scores.size() != 2
    ) {
        std::cerr
            << "Unexpected clash score count\n";

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_protein_grid(
            device_grid
        );

        return 1;
    }


    constexpr double expected_fraction =
        1.0 / 3.0;


    for (
        std::size_t i = 0;
        i < scores.size();
        ++i
    ) {
        if (
            scores[i].valid == 0 ||
            scores[i].occupied_atoms != 1 ||
            scores[i].total_atoms != 3 ||
            std::abs(
                scores[i].fraction -
                expected_fraction
            ) > 1e-12
        ) {
            std::cerr
                << "Unexpected clash result at pose "
                << i
                << '\n';

            cuda_free_points(
                device_ligand_atoms
            );

            cuda_free_protein_grid(
                device_grid
            );

            return 1;
        }
    }


    std::cout
        << "CUDA grid clash counting passed\n";


    /*
     * Explicitly verify the serial semantic:
     *
     * the third atom is outside the grid,
     * but total_atoms remains 3 and it does
     * not count as occupied.
     */
    if (
        scores[0].total_atoms != 3 ||
        scores[0].occupied_atoms != 1
    ) {
        std::cerr
            << "CUDA out-of-grid clash semantics failed\n";

        cuda_free_points(
            device_ligand_atoms
        );

        cuda_free_protein_grid(
            device_grid
        );

        return 1;
    }


    std::cout
        << "CUDA out-of-grid clash semantics passed\n";


    cuda_free_points(
        device_ligand_atoms
    );

    cuda_free_protein_grid(
        device_grid
    );


    std::cout
        << "CUDA clash tests passed\n";


    return 0;
}