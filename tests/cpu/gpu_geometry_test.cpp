#include "cuda_geometry.hpp"

#include <GraphMol/Conformer.h>
#include <GraphMol/RWMol.h>

#include <cmath>
#include <iostream>
#include <vector>


namespace
{

bool same_point(
    const GpuPoint3& point,
    double x,
    double y,
    double z
)
{
    constexpr double tolerance = 1e-12;

    return
        std::abs(point.x - x) < tolerance &&
        std::abs(point.y - y) < tolerance &&
        std::abs(point.z - z) < tolerance;
}

} // namespace


int main()
{
    // ========================================================
    // Ligand matching-area conversion
    // ========================================================

    const std::vector<MatchingArea>
        ligand_areas = {

            {
                0,
                {1.0, 2.0, 3.0},
                InteractionKey::Hydrophobic,
                0.0
            },

            {
                1,
                {-4.0, 5.5, 6.25},
                InteractionKey::HydrogenDonor,
                0.0
            }
        };


    const auto ligand_points =
        make_gpu_ligand_matching_points(
            ligand_areas
        );


    if (
        ligand_points.size() != 2 ||
        !same_point(
            ligand_points[0],
            1.0, 2.0, 3.0
        ) ||
        !same_point(
            ligand_points[1],
            -4.0, 5.5, 6.25
        )
    ) {
        std::cerr
            << "Ligand GPU point conversion failed\n";

        return 1;
    }


    std::cout
        << "Ligand GPU point conversion passed\n";


    // ========================================================
    // Pose-target conversion
    // ========================================================

    const std::vector<PoseTarget>
        pose_targets = {

            {
                10,
                {7.0, 8.0, 9.0},
                InteractionKey::Hydrophobic
            },

            {
                11,
                {-1.5, 2.25, -3.75},
                InteractionKey::HydrogenAcceptor
            }
        };


    const auto target_points =
        make_gpu_pose_target_points(
            pose_targets
        );


    if (
        target_points.size() != 2 ||
        !same_point(
            target_points[0],
            7.0, 8.0, 9.0
        ) ||
        !same_point(
            target_points[1],
            -1.5, 2.25, -3.75
        )
    ) {
        std::cerr
            << "Pose-target GPU point conversion failed\n";

        return 1;
    }


    std::cout
        << "Pose-target GPU point conversion passed\n";


    std::cout
        << "GPU geometry conversion tests passed\n";

    // ========================================================
    // Ligand atom-coordinate conversion
    // ========================================================

    RDKit::RWMol ligand;


    ligand.addAtom(
        new RDKit::Atom(6),
        true,
        true
    );

    ligand.addAtom(
        new RDKit::Atom(8),
        true,
        true
    );


    auto* conformer =
        new RDKit::Conformer(2);


    conformer->setAtomPos(
        0,
        RDGeom::Point3D(
            1.25,
            -2.5,
            3.75
        )
    );


    conformer->setAtomPos(
        1,
        RDGeom::Point3D(
            -4.0,
            5.5,
            6.25
        )
    );


    ligand.addConformer(
        conformer,
        true
    );


    const auto ligand_atom_points =
        make_gpu_ligand_atom_points(
            ligand
        );


    if (
        ligand_atom_points.size() != 2 ||
        !same_point(
            ligand_atom_points[0],
            1.25,
            -2.5,
            3.75
        ) ||
        !same_point(
            ligand_atom_points[1],
            -4.0,
            5.5,
            6.25
        )
    ) {
        std::cerr
            << "Ligand atom GPU point conversion failed\n";

        return 1;
    }


    std::cout
        << "Ligand atom GPU point conversion passed\n";


    return 0;
}