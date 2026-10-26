#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "protein_grid.hpp"
#include "run_config.hpp"

#include <interaction/elements.hpp>
#include <interaction/find_interaction.hpp>

#include <GraphMol/Atom.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <utility>
#include <vector>


namespace
{

interaction::point3D voxel_center(
    const ProteinGrid& grid,
    int x,
    int y,
    int z)
{
    return {
        static_cast<interaction::coordinate_type>(
            grid.xmin +
            (static_cast<double>(x) + 0.5) *
            grid.voxel_size
        ),

        static_cast<interaction::coordinate_type>(
            grid.ymin +
            (static_cast<double>(y) + 0.5) *
            grid.voxel_size
        ),

        static_cast<interaction::coordinate_type>(
            grid.zmin +
            (static_cast<double>(z) + 0.5) *
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


    for (
        unsigned int atom_index = 0;
        atom_index < pocket.getNumAtoms();
        ++atom_index
    ) {
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


        for (
            int dx = -grid_radius;
            dx <= grid_radius;
            ++dx
        ) {
            for (
                int dy = -grid_radius;
                dy <= grid_radius;
                ++dy
            ) {
                for (
                    int dz = -grid_radius;
                    dz <= grid_radius;
                    ++dz
                ) {
                    const int x =
                        gx + dx;

                    const int y =
                        gy + dy;

                    const int z =
                        gz + dz;


                    if (!grid.in_bounds(
                            x,
                            y,
                            z
                        )) {
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


                    if (
                        distance_squared >
                        radius_squared
                    ) {
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


double distance_to_protein_surface(
    const interaction::point3D& point,
    const RDKit::ROMol& protein)
{
    const auto& conformer =
        protein.getConformer();


    double best_distance =
        std::numeric_limits<double>::
            infinity();


    for (
        unsigned int atom_index = 0;
        atom_index < protein.getNumAtoms();
        ++atom_index
    ) {
        const auto atom_position =
            conformer.getAtomPos(
                atom_index
            );


        const double dx =
            static_cast<double>(
                point.x
            ) -
            atom_position.x;

        const double dy =
            static_cast<double>(
                point.y
            ) -
            atom_position.y;

        const double dz =
            static_cast<double>(
                point.z
            ) -
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


        best_distance =
            std::min(
                best_distance,
                center_distance - radius
            );
    }


    return best_distance;
}


struct AccessibilityRecord
{
    std::size_t source_area_index = 0;

    InteractionKey key{};

    double radius = 0.0;

    std::size_t pocket_voxels = 0;
    std::size_t free_voxels = 0;

    double free_fraction = 0.0;

    double max_clearance =
        -std::numeric_limits<double>::
            infinity();

    bool crystal_interaction = false;
};


struct ThresholdStats
{
    double threshold = 0.0;

    std::size_t areas_below = 0;

    std::size_t crystal_areas_below = 0;
};

} // namespace


int main(int argc, char* argv[])
{
    try {

        const auto config =
            parse_run_config(
                argc,
                argv
            );


        // ====================================================
        // Molecules
        // ====================================================

        auto protein =
            load_protein(
                config.protein_path.string()
            );

        auto pocket =
            load_pocket(
                config.pocket_path.string()
            );

        auto ligand =
            load_ligand(
                config.ligand_path.string()
            );


        // ====================================================
        // Interaction areas
        // ====================================================

        const auto protein_areas =
            get_protein_areas(
                *protein
            );

        const auto ligand_areas =
            get_ligand_areas(
                *ligand
            );


        const auto protein_matching_areas =
            build_protein_matching_areas(
                protein_areas
            );

        const auto ligand_matching_areas =
            build_ligand_matching_areas(
                ligand_areas
            );


        const auto pocket_matching_areas =
            filter_matching_areas_near_pocket(
                protein_matching_areas,
                *pocket,
                config.pocket_centroid_distance
            );


        // ====================================================
        // Crystal interaction oracle
        //
        // A logical protein matching area is considered
        // crystal-linked only when:
        //
        //  - its physical protein source participates
        //    in an interaction;
        //  - its logical key matches the ligand feature
        //    involved in that interaction.
        // ====================================================

        const auto interactions =
            interaction::find_interactions(
                protein_areas,
                ligand_areas
            );


        using CrystalKey =
            std::pair<std::size_t, int>;


        std::set<CrystalKey>
            crystal_matching_areas;


        for (const auto& formed :
             interactions) {

            const std::size_t protein_source =
                formed
                    .index_protein_interaction_area;

            const std::size_t ligand_source =
                formed
                    .index_ligand_interaction_area;


            for (const auto& ligand_area :
                 ligand_matching_areas) {

                if (
                    ligand_area.source_area_index !=
                    ligand_source
                ) {
                    continue;
                }


                crystal_matching_areas.insert({
                    protein_source,
                    static_cast<int>(
                        ligand_area.key
                    )
                });
            }
        }


        // ====================================================
        // Grid
        // ====================================================

        auto grid =
            build_empty_protein_grid(
                *pocket,
                config.grid_voxel_size,
                config.grid_margin
            );


        mark_protein_occupancy(
            *protein,
            grid
        );


        // Same global region used by PoseTargets.
        constexpr double POCKET_REGION_RADIUS =
            6.5;


        std::vector<bool> pocket_region(
            grid.voxels.size(),
            false
        );


        mark_pocket_region(
            *pocket,
            grid,
            POCKET_REGION_RADIUS,
            pocket_region
        );


        // Same idea used by production PoseTarget generation:
        // clearance of a voxel is reused across areas.
        std::vector<double> clearance_cache(
            grid.voxels.size(),
            std::numeric_limits<double>::
                quiet_NaN()
        );


        // ====================================================
        // Measure each pocket matching area
        // ====================================================

        std::vector<AccessibilityRecord>
            records;


        records.reserve(
            pocket_matching_areas.size()
        );


        for (const auto& area :
             pocket_matching_areas) {

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
                radius *
                radius;


            std::size_t pocket_voxels = 0;
            std::size_t free_voxels = 0;


            double max_clearance =
                -std::numeric_limits<double>::
                    infinity();


            for (
                int dx = -grid_radius;
                dx <= grid_radius;
                ++dx
            ) {
                for (
                    int dy = -grid_radius;
                    dy <= grid_radius;
                    ++dy
                ) {
                    for (
                        int dz = -grid_radius;
                        dz <= grid_radius;
                        ++dz
                    ) {
                        const int x =
                            gx + dx;

                        const int y =
                            gy + dy;

                        const int z =
                            gz + dz;


                        if (!grid.in_bounds(
                                x,
                                y,
                                z
                            )) {
                            continue;
                        }


                        const std::size_t index =
                            grid.index(
                                x,
                                y,
                                z
                            );


                        if (!pocket_region[index])
                            continue;


                        const auto point =
                            voxel_center(
                                grid,
                                x,
                                y,
                                z
                            );


                        const double vx =
                            static_cast<double>(
                                point.x
                            ) - cx;

                        const double vy =
                            static_cast<double>(
                                point.y
                            ) - cy;

                        const double vz =
                            static_cast<double>(
                                point.z
                            ) - cz;


                        const double d2 =
                            vx * vx +
                            vy * vy +
                            vz * vz;


                        if (
                            d2 >
                            radius_squared
                        ) {
                            continue;
                        }


                        ++pocket_voxels;


                        if (
                            grid.voxels[
                                index
                            ].occupied
                        ) {
                            continue;
                        }


                        ++free_voxels;


                        double clearance =
                            clearance_cache[
                                index
                            ];


                        if (
                            std::isnan(
                                clearance
                            )
                        ) {
                            clearance =
                                distance_to_protein_surface(
                                    point,
                                    *protein
                                );


                            clearance_cache[
                                index
                            ] =
                                clearance;
                        }


                        max_clearance =
                            std::max(
                                max_clearance,
                                clearance
                            );
                    }
                }
            }


            const double free_fraction =
                pocket_voxels == 0
                    ? 0.0
                    : (
                        static_cast<double>(
                            free_voxels
                        ) /
                        static_cast<double>(
                            pocket_voxels
                        )
                    );


            const bool crystal_interaction =
                crystal_matching_areas.contains({
                    area.source_area_index,
                    static_cast<int>(
                        area.key
                    )
                });


            records.push_back({
                area.source_area_index,
                area.key,
                radius,
                pocket_voxels,
                free_voxels,
                free_fraction,
                max_clearance,
                crystal_interaction
            });
        }


        // ====================================================
        // Summary
        // ====================================================

        std::size_t crystal_linked_count =
            0;


        for (const auto& record :
             records) {

            if (
                record.crystal_interaction
            ) {
                ++crystal_linked_count;
            }
        }


        std::vector<ThresholdStats>
            thresholds{
                {0.50, 0, 0},
                {0.75, 0, 0},
                {1.00, 0, 0},
                {1.25, 0, 0},
                {1.50, 0, 0},
                {2.00, 0, 0}
            };


        for (const auto& record :
             records) {

            for (auto& threshold :
                 thresholds) {

                if (
                    record.max_clearance <
                    threshold.threshold
                ) {
                    ++threshold.areas_below;


                    if (
                        record.crystal_interaction
                    ) {
                        ++threshold
                            .crystal_areas_below;
                    }
                }
            }
        }


        std::cout
            << std::fixed
            << std::setprecision(4);


        std::cout
            << "=== AREA ACCESSIBILITY + CLEARANCE ===\n\n"

            << "Protein atoms: "
            << protein->getNumAtoms()
            << '\n'

            << "Pocket matching areas: "
            << pocket_matching_areas.size()
            << '\n'

            << "Measured areas: "
            << records.size()
            << '\n'

            << "Crystal interactions: "
            << interactions.size()
            << '\n'

            << "Crystal-linked logical areas: "
            << crystal_linked_count
            << "\n\n";


        std::cout
            << "=== CLEARANCE THRESHOLD DIAGNOSTIC ===\n"

            << "threshold"
            << "\tareas_below"
            << "\tcrystal_below"
            << "\tareas_retained"
            << '\n';


        for (const auto& threshold :
             thresholds) {

            std::cout
                << threshold.threshold
                << '\t'

                << threshold.areas_below
                << '\t'

                << threshold.crystal_areas_below
                << '\t'

                << (
                    records.size() -
                    threshold.areas_below
                )
                << '\n';
        }


        // ====================================================
        // Worst areas first
        // ====================================================

        std::sort(
            records.begin(),
            records.end(),
            [](
                const AccessibilityRecord& a,
                const AccessibilityRecord& b
            ) {
                return
                    a.max_clearance <
                    b.max_clearance;
            }
        );


        std::cout
            << "\n=== 40 LOWEST MAX-CLEARANCE AREAS ===\n"

            << "source"
            << "\tkey"
            << "\tradius"
            << "\tfree_pct"
            << "\tmax_clearance"
            << "\tcrystal"
            << '\n';


        const std::size_t count =
            std::min<std::size_t>(
                40,
                records.size()
            );


        for (
            std::size_t i = 0;
            i < count;
            ++i
        ) {
            const auto& record =
                records[i];


            std::cout
                << record.source_area_index
                << '\t'

                << static_cast<int>(
                    record.key
                )
                << '\t'

                << record.radius
                << '\t'

                << (
                    record.free_fraction *
                    100.0
                )
                << '\t'

                << record.max_clearance
                << '\t'

                << (
                    record.crystal_interaction
                        ? "YES"
                        : "NO"
                )
                << '\n';
        }


        return 0;
    }
    catch (const std::exception& error) {

        std::cerr
            << "Error: "
            << error.what()
            << '\n';

        return 1;
    }
}
