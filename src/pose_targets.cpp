#include "pose_targets.hpp"

#include <interaction/elements.hpp>

#include <GraphMol/Atom.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>


namespace
{

double atom_vdw_radius(
    const RDKit::Atom& atom)
{
    const auto element =
        interaction::parse_chemical_element(
            atom.getSymbol()
        );

    if (element.has_value()) {
        return interaction::get(
            element.value()
        ).vdw_radius;
    }

    return 1.7;
}


// Signed clearance: min(distance to atom center - VDW radius), in angstroms.
// This tests all protein atoms, with the same 1.7 A fallback for unknown elements.
double distance_to_protein_surface(
    const interaction::point3D& point,
    const RDKit::ROMol& protein)
{
    const auto& conformer =
        protein.getConformer();

    double best_distance =
        std::numeric_limits<double>::infinity();


    for (unsigned int atom_index = 0;
         atom_index < protein.getNumAtoms();
         ++atom_index) {

        const auto atom_position =
            conformer.getAtomPos(
                atom_index
            );


        const double dx =
            static_cast<double>(point.x) -
            atom_position.x;

        const double dy =
            static_cast<double>(point.y) -
            atom_position.y;

        const double dz =
            static_cast<double>(point.z) -
            atom_position.z;


        const double center_distance =
            std::sqrt(
                dx * dx +
                dy * dy +
                dz * dz
            );


        const double radius =
            atom_vdw_radius(
                *protein.getAtomWithIdx(
                    atom_index
                )
            );


        const double surface_distance =
            center_distance -
            radius;


        best_distance =
            std::min(
                best_distance,
                surface_distance
            );
    }


    return best_distance;
}


interaction::point3D voxel_center(
    const ProteinGrid& grid,
    int x,
    int y,
    int z)
{
    return {
        static_cast<
            interaction::coordinate_type
        >(
            grid.xmin +
            (
                static_cast<double>(x) +
                0.5
            ) *
            grid.voxel_size
        ),

        static_cast<
            interaction::coordinate_type
        >(
            grid.ymin +
            (
                static_cast<double>(y) +
                0.5
            ) *
            grid.voxel_size
        ),

        static_cast<
            interaction::coordinate_type
        >(
            grid.zmin +
            (
                static_cast<double>(z) +
                0.5
            ) *
            grid.voxel_size
        )
    };
}


void mark_pocket_region(
    const RDKit::ROMol& pocket,
    const ProteinGrid& grid,
    double region_radius,
    std::vector<bool>& region)
{
    const auto& conformer =
        pocket.getConformer();


    const double radius_squared =
        region_radius *
        region_radius;


    const int grid_radius =
        static_cast<int>(
            std::ceil(
                region_radius /
                grid.voxel_size
            )
        ) + 1;


    for (unsigned int atom_index = 0;
         atom_index < pocket.getNumAtoms();
         ++atom_index) {

        const auto atom_position =
            conformer.getAtomPos(
                atom_index
            );


        const int gx =
            grid.to_grid(
                atom_position.x,
                grid.xmin
            );

        const int gy =
            grid.to_grid(
                atom_position.y,
                grid.ymin
            );

        const int gz =
            grid.to_grid(
                atom_position.z,
                grid.zmin
            );


        for (int dx = -grid_radius;
             dx <= grid_radius;
             ++dx) {

            for (int dy = -grid_radius;
                 dy <= grid_radius;
                 ++dy) {

                for (int dz = -grid_radius;
                     dz <= grid_radius;
                     ++dz) {

                    const int x =
                        gx + dx;

                    const int y =
                        gy + dy;

                    const int z =
                        gz + dz;


                    if (!grid.in_bounds(
                            x,
                            y,
                            z)) {

                        continue;
                    }


                    const auto point =
                        voxel_center(
                            grid,
                            x,
                            y,
                            z
                        );


                    const double ddx =
                        static_cast<double>(
                            point.x
                        ) -
                        atom_position.x;

                    const double ddy =
                        static_cast<double>(
                            point.y
                        ) -
                        atom_position.y;

                    const double ddz =
                        static_cast<double>(
                            point.z
                        ) -
                        atom_position.z;


                    const double distance_squared =
                        ddx * ddx +
                        ddy * ddy +
                        ddz * ddz;


                    if (distance_squared >
                        radius_squared) {

                        continue;
                    }


                    region[
                        grid.index(
                            x,
                            y,
                            z
                        )
                    ] = true;
                }
            }
        }
    }
}

} // namespace

std::size_t prune_pose_targets(
    std::vector<PoseTarget>& targets,
    const RDKit::ROMol& protein,
    std::optional<double> minimum_clearance)
{
    if (!minimum_clearance) return 0;
    if (!std::isfinite(*minimum_clearance) || *minimum_clearance < 0.0) {
        throw std::invalid_argument("Minimum target clearance must be finite and non-negative");
    }

    const auto before = targets.size();
    targets.erase(std::remove_if(targets.begin(), targets.end(),
        [&](const PoseTarget& target) {
            return !(distance_to_protein_surface(target.position, protein) >= *minimum_clearance);
        }), targets.end());
    return before - targets.size();
}

std::vector<PoseTarget> generate_pose_targets(
    const std::vector<MatchingArea>& protein_areas,
    const ProteinGrid& grid,
    const RDKit::ROMol& protein,
    const RDKit::ROMol& pocket,
    double pocket_region_radius)
{
    std::vector<PoseTarget> targets;

    targets.reserve(
        protein_areas.size()
    );


    // Allowed region is the union of spheres around pocket atoms. Its radius
    // (normally 6.5 A) is independent of interaction radii and grid padding.
    std::vector<bool> pocket_region(
        grid.voxels.size(),
        false
    );


    mark_pocket_region(
        pocket,
        grid,
        pocket_region_radius,
        pocket_region
    );


    // Logical areas can revisit a voxel; cache its protein-wide clearance once.
    std::vector<double> clearance_cache(
        grid.voxels.size(),
        std::numeric_limits<double>::quiet_NaN()
    );


    for (const auto& area :
         protein_areas) {

        // Each logical area keeps its own interaction radius.
        const double radius =
            area.interaction_radius;


        if (radius <= 0.0)
            continue;


        const double cx =
            static_cast<double>(
                area.centroid.x
            );

        const double cy =
            static_cast<double>(
                area.centroid.y
            );

        const double cz =
            static_cast<double>(
                area.centroid.z
            );


        const int gx =
            grid.to_grid(
                cx,
                grid.xmin
            );

        const int gy =
            grid.to_grid(
                cy,
                grid.ymin
            );

        const int gz =
            grid.to_grid(
                cz,
                grid.zmin
            );


        const int grid_radius =
            static_cast<int>(
                std::ceil(
                    radius /
                    grid.voxel_size
                )
            ) + 1;


        const double radius_squared =
            radius * radius;


        bool found = false;


        interaction::point3D best_point{};


        double best_clearance =
            -std::numeric_limits<double>::infinity();


        double best_distance_squared =
            std::numeric_limits<double>::infinity();


        for (int dx = -grid_radius;
             dx <= grid_radius;
             ++dx) {

            for (int dy = -grid_radius;
                 dy <= grid_radius;
                 ++dy) {

                for (int dz = -grid_radius;
                     dz <= grid_radius;
                     ++dz) {

                    const int x =
                        gx + dx;

                    const int y =
                        gy + dy;

                    const int z =
                        gz + dz;


                    if (!grid.in_bounds(
                            x,
                            y,
                            z)) {

                        continue;
                    }


                    const std::size_t voxel_index =
                        grid.index(
                            x,
                            y,
                            z
                        );


                    if (
                        grid.voxels[
                            voxel_index
                        ].occupied
                    ) {
                        continue;
                    }


                    if (!pocket_region[
                            voxel_index
                        ]) {

                        continue;
                    }


                    const auto candidate =
                        voxel_center(
                            grid,
                            x,
                            y,
                            z
                        );


                    const double vx =
                        static_cast<double>(
                            candidate.x
                        ) -
                        cx;

                    const double vy =
                        static_cast<double>(
                            candidate.y
                        ) -
                        cy;

                    const double vz =
                        static_cast<double>(
                            candidate.z
                        ) -
                        cz;


                    const double distance_squared =
                        vx * vx +
                        vy * vy +
                        vz * vz;


                    // Test the voxel center against this area's radius, not 6.5 A.
                    if (distance_squared >
                        radius_squared) {

                        continue;
                    }


                    double clearance =
                        clearance_cache[
                            voxel_index
                        ];


                    if (std::isnan(
                            clearance)) {

                        clearance =
                            distance_to_protein_surface(
                                candidate,
                                protein
                            );


                        clearance_cache[
                            voxel_index
                        ] =
                            clearance;
                    }


                    // Prefer greater clearance; also accept a voxel closer to the
                    // area centroid when clearance differs by less than 1e-6 A.
                    // Remaining ties retain the first voxel in dx/dy/dz traversal order.
                    const bool better =
                        !found
                        ||
                        clearance >
                            best_clearance
                        ||
                        (
                            std::abs(
                                clearance -
                                best_clearance
                            ) < 1e-6
                            &&
                            distance_squared <
                                best_distance_squared
                        );


                    if (!better)
                        continue;


                    found = true;

                    best_point =
                        candidate;

                    best_clearance =
                        clearance;

                    best_distance_squared =
                        distance_squared;
                }
            }
        }


        // No centroid fallback: an area with no valid free voxel produces no target.
        if (!found)
            continue;


        targets.push_back({
            area.source_area_index,
            best_point,
            area.key
        });
    }


    return targets;
}

std::vector<MatchingArea>
filter_matching_areas_in_pocket(
    const std::vector<MatchingArea>& matching_areas,
    const std::vector<interaction::interaction_area>& protein_areas,
    const RDKit::ROMol& pocket)
{
    constexpr double TOLERANCE = 1e-3;
    constexpr double TOLERANCE_SQUARED =
        TOLERANCE * TOLERANCE;

    std::vector<MatchingArea> result;

    result.reserve(
        matching_areas.size()
    );

    const auto& pocket_conformer =
        pocket.getConformer();


    for (const auto& matching_area :
         matching_areas) {

        if (
            matching_area.source_area_index >=
            protein_areas.size()
        ) {
            continue;
        }


        const auto& physical_area =
            protein_areas[
                matching_area.source_area_index
            ];


        bool all_atoms_in_pocket = true;


        for (std::size_t area_atom = 0;
             area_atom < physical_area.num_atoms;
             ++area_atom) {

            const double ax =
                static_cast<double>(
                    physical_area.x[
                        area_atom
                    ]
                );

            const double ay =
                static_cast<double>(
                    physical_area.y[
                        area_atom
                    ]
                );

            const double az =
                static_cast<double>(
                    physical_area.z[
                        area_atom
                    ]
                );


            bool atom_found = false;


            for (unsigned int pocket_atom = 0;
                 pocket_atom < pocket.getNumAtoms();
                 ++pocket_atom) {

                const auto position =
                    pocket_conformer.getAtomPos(
                        pocket_atom
                    );


                const double dx =
                    ax - position.x;

                const double dy =
                    ay - position.y;

                const double dz =
                    az - position.z;


                const double distance_squared =
                    dx * dx +
                    dy * dy +
                    dz * dz;


                if (
                    distance_squared <=
                    TOLERANCE_SQUARED
                ) {
                    atom_found = true;
                    break;
                }
            }


            if (!atom_found) {

                all_atoms_in_pocket = false;
                break;
            }
        }


        if (all_atoms_in_pocket) {

            result.push_back(
                matching_area
            );
        }
    }


    return result;
}
