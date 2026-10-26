#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "run_config.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>


namespace
{

using Clock =
    std::chrono::steady_clock;


double seconds(
    const Clock::time_point& start,
    const Clock::time_point& end)
{
    return
        std::chrono::duration<double>(
            end - start
        ).count();
}

} // namespace


int main(int argc, char* argv[])
{
    try {

        const auto config =
            parse_run_config(
                argc,
                argv
            );


        std::cout
            << std::fixed
            << std::setprecision(6);


        const auto total_start =
            Clock::now();


        // ====================================================
        // 1. Load protein
        // ====================================================

        auto start =
            Clock::now();

        auto protein =
            load_protein(
                config.protein_path.string()
            );

        auto end =
            Clock::now();

        const double load_protein_seconds =
            seconds(start, end);


        // ====================================================
        // 2. Load pocket
        // ====================================================

        start =
            Clock::now();

        auto pocket =
            load_pocket(
                config.pocket_path.string()
            );

        end =
            Clock::now();

        const double load_pocket_seconds =
            seconds(start, end);


        // ====================================================
        // 3. Protein physical interaction areas
        // ====================================================

        start =
            Clock::now();

        const auto protein_areas =
            get_protein_areas(
                *protein
            );

        end =
            Clock::now();

        const double protein_areas_seconds =
            seconds(start, end);


        // ====================================================
        // 4. Physical areas belonging to pocket
        // ====================================================

        start =
            Clock::now();

        const auto pocket_protein_areas =
            filter_protein_areas_in_pocket(
                protein_areas,
                *pocket
            );

        end =
            Clock::now();

        const double pocket_area_filter_seconds =
            seconds(start, end);


        // ====================================================
        // 5. Logical protein matching areas
        // ====================================================

        start =
            Clock::now();

        const auto protein_matching_areas =
            build_protein_matching_areas(
                protein_areas
            );

        end =
            Clock::now();

        const double matching_areas_seconds =
            seconds(start, end);


        // ====================================================
        // 6. Empty grid
        // ====================================================

        start =
            Clock::now();

        auto protein_grid =
            build_empty_protein_grid(
                *pocket,
                config.grid_voxel_size,
                config.grid_margin
            );

        end =
            Clock::now();

        const double empty_grid_seconds =
            seconds(start, end);


        // ====================================================
        // 7. Protein occupancy
        // ====================================================

        start =
            Clock::now();

        mark_protein_occupancy(
            *protein,
            protein_grid
        );

        end =
            Clock::now();

        const double occupancy_seconds =
            seconds(start, end);


        // ====================================================
        // 8. Interaction regions
        // ====================================================

        start =
            Clock::now();

        mark_interaction_regions(
            protein_matching_areas,
            protein_grid
        );

        end =
            Clock::now();

        const double interaction_regions_seconds =
            seconds(start, end);


        // ====================================================
        // 9. Matching areas near pocket
        // ====================================================

        start =
            Clock::now();

        const auto pocket_matching_areas =
            filter_matching_areas_near_pocket(
                protein_matching_areas,
                *pocket,
                config.pocket_centroid_distance
            );

        end =
            Clock::now();

        const double matching_filter_seconds =
            seconds(start, end);


        // ====================================================
        // 10. Pose targets
        // ====================================================

        start =
            Clock::now();

        const auto pose_targets =
            generate_pose_targets(
                pocket_matching_areas,
                protein_grid,
                *protein,
                *pocket
            );

        end =
            Clock::now();

        const double pose_targets_seconds =
            seconds(start, end);


        const auto total_end =
            Clock::now();


        // ====================================================
        // Output
        // ====================================================

        std::cout
            << "=== OFFLINE PROFILE ===\n\n"

            << "Protein atoms: "
            << protein->getNumAtoms()
            << '\n'

            << "Pocket atoms: "
            << pocket->getNumAtoms()
            << '\n'

            << "Protein physical areas: "
            << protein_areas.size()
            << '\n'

            << "Pocket physical areas: "
            << pocket_protein_areas.size()
            << '\n'

            << "Protein matching areas: "
            << protein_matching_areas.size()
            << '\n'

            << "Pocket matching areas: "
            << pocket_matching_areas.size()
            << '\n'

            << "Pose targets: "
            << pose_targets.size()
            << '\n'

            << "Grid: "
            << protein_grid.nx
            << " x "
            << protein_grid.ny
            << " x "
            << protein_grid.nz
            << '\n'

            << "Grid voxels: "
            << protein_grid.voxels.size()
            << "\n\n";


        std::cout
            << "=== TIMINGS ===\n"

            << "load_protein:              "
            << load_protein_seconds
            << " s\n"

            << "load_pocket:               "
            << load_pocket_seconds
            << " s\n"

            << "get_protein_areas:          "
            << protein_areas_seconds
            << " s\n"

            << "filter_physical_pocket:     "
            << pocket_area_filter_seconds
            << " s\n"

            << "build_matching_areas:       "
            << matching_areas_seconds
            << " s\n"

            << "build_empty_grid:           "
            << empty_grid_seconds
            << " s\n"

            << "mark_protein_occupancy:     "
            << occupancy_seconds
            << " s\n"

            << "mark_interaction_regions:   "
            << interaction_regions_seconds
            << " s\n"

            << "filter_matching_pocket:     "
            << matching_filter_seconds
            << " s\n"

            << "generate_pose_targets:      "
            << pose_targets_seconds
            << " s\n"

            << "----------------------------------------\n"

            << "TOTAL OFFLINE:              "
            << seconds(
                total_start,
                total_end
            )
            << " s\n";


        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "Error: "
            << e.what()
            << '\n';

        return 1;
    }
}
