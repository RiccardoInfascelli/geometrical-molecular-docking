#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "run_config.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>


namespace
{

constexpr double PI =
    3.14159265358979323846;


double point_distance(
    const interaction::point3D& a,
    const interaction::point3D& b)
{
    const double dx =
        static_cast<double>(a.x) -
        static_cast<double>(b.x);

    const double dy =
        static_cast<double>(a.y) -
        static_cast<double>(b.y);

    const double dz =
        static_cast<double>(a.z) -
        static_cast<double>(b.z);

    return std::sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    );
}


const MatchingArea* find_matching_area(
    const std::vector<MatchingArea>& areas,
    std::size_t source_area_index,
    InteractionKey key)
{
    for (const auto& area : areas) {

        if (
            area.source_area_index ==
                source_area_index &&
            area.key == key
        ) {
            return &area;
        }
    }

    return nullptr;
}


double vector_angle_degrees(
    const interaction::point3D& origin,
    const interaction::point3D& a,
    const interaction::point3D& b)
{
    const double ax =
        static_cast<double>(a.x) -
        static_cast<double>(origin.x);

    const double ay =
        static_cast<double>(a.y) -
        static_cast<double>(origin.y);

    const double az =
        static_cast<double>(a.z) -
        static_cast<double>(origin.z);


    const double bx =
        static_cast<double>(b.x) -
        static_cast<double>(origin.x);

    const double by =
        static_cast<double>(b.y) -
        static_cast<double>(origin.y);

    const double bz =
        static_cast<double>(b.z) -
        static_cast<double>(origin.z);


    const double norm_a =
        std::sqrt(
            ax * ax +
            ay * ay +
            az * az
        );

    const double norm_b =
        std::sqrt(
            bx * bx +
            by * by +
            bz * bz
        );


    if (
        norm_a < 1e-12 ||
        norm_b < 1e-12
    ) {
        return 0.0;
    }


    double cosine =
        (
            ax * bx +
            ay * by +
            az * bz
        ) /
        (
            norm_a *
            norm_b
        );


    cosine =
        std::clamp(
            cosine,
            -1.0,
            1.0
        );


    return
        std::acos(cosine) *
        180.0 /
        PI;
}


struct OracleRow
{
    std::size_t protein_source = 0;
    std::size_t ligand_source = 0;

    InteractionKey key{};

    double radius = 0.0;

    double oracle_shift = 0.0;
    double actual_shift = 0.0;

    double shift_length_error = 0.0;

    double target_position_error = 0.0;

    double direction_error_degrees = 0.0;
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
        // Interaction / matching areas
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


        // ====================================================
        // Same grid / PoseTargets as production
        // ====================================================

        auto protein_grid =
            build_empty_protein_grid(
                *pocket,
                config.grid_voxel_size,
                config.grid_margin
            );


        mark_protein_occupancy(
            *protein,
            protein_grid
        );


        mark_interaction_regions(
            protein_matching_areas,
            protein_grid
        );


        const auto pocket_matching_areas =
            filter_matching_areas_near_pocket(
                protein_matching_areas,
                *pocket,
                config.pocket_centroid_distance
            );


        const auto pose_targets =
            generate_pose_targets(
                pocket_matching_areas,
                protein_grid,
                *protein,
                *pocket
            );


        // ====================================================
        // Geometric oracle
        //
        // For every protein PoseTarget:
        //
        // P = original protein matching-area centroid
        // T = PoseTarget chosen by our algorithm
        // L = nearest compatible crystal ligand area
        //     with same key and inside interaction radius
        //
        // Oracle vector = P -> L
        // Actual vector = P -> T
        // ====================================================

        std::vector<OracleRow> rows;

        std::size_t no_compatible_ligand_area = 0;

        std::size_t improved_vs_original = 0;
        std::size_t worsened_vs_original = 0;


        double oracle_shift_sum = 0.0;
        double actual_shift_sum = 0.0;

        double position_error_sum = 0.0;
        double direction_error_sum = 0.0;

        double absolute_shift_length_error_sum =
            0.0;


        std::size_t angle_lt_15 = 0;
        std::size_t angle_15_30 = 0;
        std::size_t angle_30_60 = 0;
        std::size_t angle_ge_60 = 0;


        for (const auto& target :
             pose_targets) {

            const auto* protein_area =
                find_matching_area(
                    pocket_matching_areas,
                    target.source_area_index,
                    target.key
                );


            if (protein_area == nullptr)
                continue;


            const MatchingArea*
                best_ligand_area =
                    nullptr;


            double best_distance =
                std::numeric_limits<
                    double
                >::infinity();


            // --------------------------------------------
            // Find nearest crystal ligand area
            // compatible with this logical protein area.
            // --------------------------------------------

            for (const auto& ligand_area :
                 ligand_matching_areas) {

                if (
                    ligand_area.key !=
                    protein_area->key
                ) {
                    continue;
                }


                const double distance =
                    point_distance(
                        protein_area->centroid,
                        ligand_area.centroid
                    );


                if (
                    distance >
                    protein_area
                        ->interaction_radius
                ) {
                    continue;
                }


                if (
                    distance <
                    best_distance
                ) {

                    best_distance =
                        distance;

                    best_ligand_area =
                        &ligand_area;
                }
            }


            if (
                best_ligand_area ==
                nullptr
            ) {

                ++no_compatible_ligand_area;
                continue;
            }


            const double oracle_shift =
                point_distance(
                    protein_area->centroid,
                    best_ligand_area->centroid
                );


            const double actual_shift =
                point_distance(
                    protein_area->centroid,
                    target.position
                );


            const double target_position_error =
                point_distance(
                    target.position,
                    best_ligand_area->centroid
                );


            const double direction_error =
                vector_angle_degrees(
                    protein_area->centroid,
                    best_ligand_area->centroid,
                    target.position
                );


            const double shift_length_error =
                actual_shift -
                oracle_shift;


            rows.push_back({
                target.source_area_index,
                best_ligand_area
                    ->source_area_index,

                target.key,

                protein_area
                    ->interaction_radius,

                oracle_shift,
                actual_shift,

                shift_length_error,

                target_position_error,

                direction_error
            });


            oracle_shift_sum +=
                oracle_shift;

            actual_shift_sum +=
                actual_shift;

            position_error_sum +=
                target_position_error;

            direction_error_sum +=
                direction_error;

            absolute_shift_length_error_sum +=
                std::abs(
                    shift_length_error
                );


            // Original centroid error is oracle_shift.
            // If T is closer to crystal L than P was,
            // relocation helped geometrically.
            if (
                target_position_error <
                oracle_shift
            ) {
                ++improved_vs_original;
            }
            else {
                ++worsened_vs_original;
            }


            if (direction_error < 15.0)
                ++angle_lt_15;

            else if (direction_error < 30.0)
                ++angle_15_30;

            else if (direction_error < 60.0)
                ++angle_30_60;

            else
                ++angle_ge_60;
        }


        // Worst position errors first.
        std::sort(
            rows.begin(),
            rows.end(),
            [](
                const OracleRow& a,
                const OracleRow& b
            ) {
                return
                    a.target_position_error >
                    b.target_position_error;
            }
        );


        const double n =
            static_cast<double>(
                rows.size()
            );


        std::cout
            << std::fixed
            << std::setprecision(3);


        std::cout
            << "=== GEOMETRIC ORACLE TARGET DIAGNOSTIC ===\n"

            << "Protein matching areas: "
            << protein_matching_areas.size()
            << '\n'

            << "Pocket matching areas: "
            << pocket_matching_areas.size()
            << '\n'

            << "Pose targets: "
            << pose_targets.size()
            << '\n'

            << "Compatible crystal correspondences: "
            << rows.size()
            << " / "
            << pose_targets.size()
            << '\n'

            << "No compatible crystal ligand area: "
            << no_compatible_ligand_area
            << "\n\n";


        if (!rows.empty()) {

            std::cout
                << "Mean oracle shift: "
                << oracle_shift_sum / n
                << " A\n"

                << "Mean actual shift: "
                << actual_shift_sum / n
                << " A\n"

                << "Mean |shift length error|: "
                << absolute_shift_length_error_sum / n
                << " A\n"

                << "Mean target position error: "
                << position_error_sum / n
                << " A\n"

                << "Mean direction error: "
                << direction_error_sum / n
                << " deg\n\n"

                << "Relocation improves crystal geometry: "
                << improved_vs_original
                << " / "
                << rows.size()
                << '\n'

                << "Relocation worsens crystal geometry: "
                << worsened_vs_original
                << " / "
                << rows.size()
                << "\n\n"

                << "Direction error distribution:\n"

                << "  < 15 deg:  "
                << angle_lt_15
                << '\n'

                << "  15-30 deg: "
                << angle_15_30
                << '\n'

                << "  30-60 deg: "
                << angle_30_60
                << '\n'

                << "  >= 60 deg: "
                << angle_ge_60
                << '\n';
        }


        std::cout
            << "\n=== CORRESPONDENCES "
               "(worst position error first) ===\n"

            << "protein"
            << "\tligand"
            << "\tkey"
            << "\tradius"
            << "\toracle"
            << "\tactual"
            << "\tshift_err"
            << "\tpos_err"
            << "\tangle_deg"
            << '\n';


        for (const auto& row :
             rows) {

            std::cout
                << row.protein_source
                << '\t'

                << row.ligand_source
                << '\t'

                << static_cast<int>(
                    row.key
                )
                << '\t'

                << row.radius
                << '\t'

                << row.oracle_shift
                << '\t'

                << row.actual_shift
                << '\t'

                << row.shift_length_error
                << '\t'

                << row.target_position_error
                << '\t'

                << row.direction_error_degrees
                << '\n';
        }


        return 0;
    }
    catch (
        const std::exception& error
    ) {

        std::cerr
            << "Error: "
            << error.what()
            << '\n';

        return 1;
    }
}
