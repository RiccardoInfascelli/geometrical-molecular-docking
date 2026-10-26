#include "cuda_grid.hpp"

#include <iostream>


int main()
{
    ProteinGrid grid;


    grid.xmin = -10.0;
    grid.ymin = -20.0;
    grid.zmin = -30.0;

    grid.voxel_size = 0.5;

    grid.nx = 2;
    grid.ny = 2;
    grid.nz = 1;


    grid.voxels.resize(4);


    grid.voxels[0].occupied = true;
    grid.voxels[0].interactions =
        InteractionMask{3};


    grid.voxels[1].occupied = false;
    grid.voxels[1].interactions =
        InteractionMask{4};


    grid.voxels[2].occupied = true;
    grid.voxels[2].interactions =
        InteractionMask{0};


    grid.voxels[3].occupied = false;
    grid.voxels[3].interactions =
        InteractionMask{12};


    const auto gpu_grid =
        make_gpu_protein_grid(
            grid
        );


    if (
        !validate_gpu_protein_grid(
            grid,
            gpu_grid
        )
    ) {
        std::cerr
            << "GPU grid conversion failed\n";

        return 1;
    }


    if (
        gpu_grid.voxels.size() != 4 ||
        gpu_grid.voxels[0].occupied != 1 ||
        gpu_grid.voxels[1].occupied != 0 ||
        gpu_grid.voxels[0].interactions != 3 ||
        gpu_grid.voxels[3].interactions != 12
    ) {
        std::cerr
            << "Unexpected GPU grid contents\n";

        return 1;
    }


    std::cout
        << "GPU protein grid conversion passed\n";

    std::cout
        << "voxels: "
        << gpu_grid.voxels.size()
        << '\n';


    return 0;
}