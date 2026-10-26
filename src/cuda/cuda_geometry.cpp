#include "cuda_geometry.hpp"


std::vector<GpuPoint3>
make_gpu_ligand_matching_points(
    const std::vector<MatchingArea>& areas
)
{
    std::vector<GpuPoint3> points;

    points.reserve(
        areas.size()
    );


    for (const auto& area : areas) {

        points.push_back({
            static_cast<double>(
                area.centroid.x
            ),

            static_cast<double>(
                area.centroid.y
            ),

            static_cast<double>(
                area.centroid.z
            )
        });
    }


    return points;
}


std::vector<GpuPoint3>
make_gpu_pose_target_points(
    const std::vector<PoseTarget>& targets
)
{
    std::vector<GpuPoint3> points;

    points.reserve(
        targets.size()
    );


    for (const auto& target : targets) {

        points.push_back({
            static_cast<double>(
                target.position.x
            ),

            static_cast<double>(
                target.position.y
            ),

            static_cast<double>(
                target.position.z
            )
        });
    }


    return points;
}

std::vector<GpuPoint3>
make_gpu_ligand_atom_points(
    const RDKit::ROMol& ligand
)
{
    std::vector<GpuPoint3> points;

    points.reserve(
        ligand.getNumAtoms()
    );


    const auto& conformer =
        ligand.getConformer();


    for (
        unsigned int atom_index = 0;
        atom_index < ligand.getNumAtoms();
        ++atom_index
    ) {
        const auto position =
            conformer.getAtomPos(
                atom_index
            );


        points.push_back({
            static_cast<double>(
                position.x
            ),

            static_cast<double>(
                position.y
            ),

            static_cast<double>(
                position.z
            )
        });
    }


    return points;
}