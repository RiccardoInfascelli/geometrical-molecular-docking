#include "geometry.hpp"
#include "interaction_areas.hpp"
#include "matching_areas.hpp"
#include "molecule_io.hpp"
#include "pose_generation.hpp"
#include "pose_targets.hpp"
#include "protein_grid.hpp"
#include "run_config.hpp"
#include "triplet_generation.hpp"

#include <interaction/elements.hpp>

#include <GraphMol/Atom.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>


namespace
{

using Clock = std::chrono::steady_clock;

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
        std::numeric_limits<double>::infinity();


    for (
        unsigned int atom_index = 0;
        atom_index < protein.getNumAtoms();
        ++atom_index
    ) {
        const auto position =
            conformer.getAtomPos(
                atom_index
            );

        const double dx =
            static_cast<double>(point.x) -
            position.x;

        const double dy =
            static_cast<double>(point.y) -
            position.y;

        const double dz =
            static_cast<double>(point.z) -
            position.z;

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


std::size_t threshold_index(
    const std::array<double, 4>& thresholds,
    double wanted)
{
    for (
        std::size_t i = 0;
        i < thresholds.size();
        ++i
    ) {
        if (
            std::abs(
                thresholds[i] -
                wanted
            ) < 1e-9
        ) {
            return i;
        }
    }

    throw std::runtime_error(
        "Requested clash threshold not found"
    );
}

} // namespace


int main(int argc, char* argv[])
{
    try {

        bool targets_only = false;
        std::optional<double> compare_clearance;
        std::vector<char*> config_arguments{argv[0]};
        for (int i = 1; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--targets-only") {
                targets_only = true;
            } else if (std::string_view(argv[i]) == "--compare-clearance") {
                if (compare_clearance.has_value()) {
                    throw std::runtime_error("--compare-clearance may only be specified once");
                }
                if (i + 1 >= argc) {
                    throw std::runtime_error("--compare-clearance requires a threshold in angstroms");
                }
                const std::string value = argv[++i];
                std::size_t parsed = 0;
                try {
                    compare_clearance = std::stod(value, &parsed);
                } catch (const std::exception&) {
                    throw std::runtime_error("--compare-clearance requires a finite, non-negative number");
                }
                if (parsed != value.size() || !std::isfinite(*compare_clearance) ||
                    *compare_clearance < 0.0) {
                    throw std::runtime_error("--compare-clearance requires a finite, non-negative number");
                }
            } else {
                config_arguments.push_back(argv[i]);
            }
        }
        if (targets_only && compare_clearance.has_value()) {
            throw std::runtime_error("--targets-only cannot be combined with --compare-clearance");
        }
        const auto config_argument_count =
            static_cast<int>(config_arguments.size());
        config_arguments.push_back(nullptr);

        const auto config =
            parse_run_config(
                config_argument_count,
                config_arguments.data()
            );


        // ====================================================
        // Same target preparation as production
        // ====================================================

        const auto protein_prep_start = compare_clearance ? Clock::now() : Clock::time_point{};

        auto protein =
            load_protein(
                config.protein_path.string()
            );

        auto pocket =
            load_pocket(
                config.pocket_path.string()
            );


        const auto protein_areas =
            get_protein_areas(
                *protein
            );


        const auto protein_matching_areas =
            build_protein_matching_areas(
                protein_areas
            );


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

        const double protein_prep_seconds = compare_clearance
            ? std::chrono::duration<double>(Clock::now() - protein_prep_start).count()
            : 0.0;


        // ====================================================
        // Clearance of each generated PoseTarget
        //
        // generate_pose_targets() already chose the
        // maximum-clearance voxel for each area.
        // ====================================================

        const auto clearance_start = compare_clearance ? Clock::now() : Clock::time_point{};

        std::vector<double>
            target_clearances;

        target_clearances.reserve(
            pose_targets.size()
        );


        for (const auto& target :
             pose_targets) {

            target_clearances.push_back(
                distance_to_protein_surface(
                    target.position,
                    *protein
                )
            );
        }

        const double clearance_seconds = compare_clearance
            ? std::chrono::duration<double>(Clock::now() - clearance_start).count()
            : 0.0;


        // Inspect the actual selected targets before reading any ligand or
        // building an index. Disabled pruning is distinct from a zero threshold.
        if (targets_only) {
            std::cout << std::fixed << std::setprecision(9)
                      << "=== TARGET CLEARANCE CHECK ===\n"
                      << "Complex: " << config.complex_id << '\n'
                      << "Protein atoms: " << protein->getNumAtoms() << '\n'
                      << "Pocket matching areas: " << pocket_matching_areas.size() << '\n'
                      << "Baseline pose targets: " << pose_targets.size() << '\n';

            if (!target_clearances.empty()) {
                const auto bounds = std::minmax_element(
                    target_clearances.begin(), target_clearances.end());
                std::cout << "Minimum target clearance (A): " << *bounds.first << '\n'
                          << "Maximum target clearance (A): " << *bounds.second << '\n';
            } else {
                std::cout << "Target clearance range: unavailable (no targets)\n";
            }

            std::cout << "min_clearance_A\ttargets_before\ttargets_retained\ttargets_removed\n"
                      << "disabled\t" << pose_targets.size() << '\t'
                      << pose_targets.size() << "\t0\n";
            for (const double threshold : {0.0, 0.75, 1.25}) {
                const auto retained = static_cast<std::size_t>(std::count_if(
                    target_clearances.begin(), target_clearances.end(),
                    [threshold](double clearance) { return clearance >= threshold; }));
                std::cout << threshold << '\t' << pose_targets.size() << '\t'
                          << retained << '\t' << pose_targets.size() - retained << '\n';
            }
            return 0;
        }


        // ====================================================
        // Ligand preparation - identical for every experiment
        // ====================================================

        const auto ligand_prep_start = compare_clearance ? Clock::now() : Clock::time_point{};

        auto ligand =
            load_ligand(
                config.ligand_path.string()
            );


        const auto ligand_areas =
            get_ligand_areas(
                *ligand
            );


        const auto ligand_matching_areas =
            build_ligand_matching_areas(
                ligand_areas
            );


        const double ligand_diameter =
            compute_ligand_diameter(
                ligand_areas
            );


        const auto ligand_triplets =
            generate_triplets(
                ligand_matching_areas,
                ligand_diameter
            );

        const double ligand_prep_seconds = compare_clearance
            ? std::chrono::duration<double>(Clock::now() - ligand_prep_start).count()
            : 0.0;


        // ====================================================
        // Clash indices
        // ====================================================

        const std::size_t clash_5_index =
            threshold_index(
                config.clash_thresholds,
                0.05
            );


        const std::size_t clash_10_index =
            threshold_index(
                config.clash_thresholds,
                0.10
            );


        // ====================================================
        // Clearance pruning sweep
        // ====================================================

        constexpr std::array<double, 5>
            CLEARANCE_THRESHOLDS{
                0.00,
                0.75,
                1.00,
                1.25,
                1.50
            };

        // No threshold means no filtering at all, including negative clearances.
        std::vector<std::optional<double>> thresholds;
        if (compare_clearance) {
            thresholds = {std::nullopt, compare_clearance};
        } else {
            for (const double threshold : CLEARANCE_THRESHOLDS) {
                thresholds.push_back(threshold);
            }
        }


        std::cout
            << std::fixed
            << std::setprecision(6);


        std::cout
            << "=== CLEARANCE PRUNING EXPERIMENT ===\n\n"

            << "Protein atoms: "
            << protein->getNumAtoms()
            << '\n'

            << "Pocket matching areas: "
            << pocket_matching_areas.size()
            << '\n'

            << "Baseline pose targets: "
            << pose_targets.size()
            << '\n'

            << "Ligand areas: "
            << ligand_matching_areas.size()
            << '\n'

            << "Ligand triplets: "
            << ligand_triplets.size()
            << "\n\n";

        if (compare_clearance) {
            std::cout
                << "Complex: " << config.complex_id << '\n'
                << "Comparison: disabled vs " << *compare_clearance << " A\n"
                << "Protein index cutoff (A): " << ligand_diameter << " (ligand diameter)\n"
                << "RMSD: all input atoms, input conformer; no refit or symmetry matching.\n"
                << "Preparation is shared; its times are repeated in both rows.\n"
                << "clearance_s re-evaluates selected targets; selection is in protein_prep_s.\n"
                << "search_s includes grid support and RMSD diagnostics; index_s is separate.\n\n";
        }


        std::cout
            << "clearance"
            << "\ttargets"
            << "\tprotein_triplets"
            << "\tcandidate_matches"
            << "\ttransforms"
            << "\tsurvivors_5pct"
            << "\tbest_rmsd_5pct"
            << "\tbest_rmsd_10pct"
            << "\tindex_s"
            << "\tsearch_s";
        if (compare_clearance) {
            std::cout << "\ttargets_removed\tprotein_prep_s\tclearance_s\tligand_prep_s\tfilter_s";
        }
        std::cout << '\n';


        for (
            const auto& clearance_threshold :
            thresholds
        ) {

            const auto filter_start = compare_clearance ? Clock::now() : Clock::time_point{};

            std::vector<PoseTarget>
                filtered_targets;

            filtered_targets.reserve(
                pose_targets.size()
            );


            for (
                std::size_t i = 0;
                i < pose_targets.size();
                ++i
            ) {
                if (
                    !clearance_threshold ||
                    target_clearances[i] >= *clearance_threshold
                ) {
                    filtered_targets.push_back(
                        pose_targets[i]
                    );
                }
            }

            // Includes allocation/copy for disabled pruning and stable filtering
            // for the threshold case; neither changes target positions or order.
            const double filter_seconds = compare_clearance
                ? std::chrono::duration<double>(Clock::now() - filter_start).count()
                : 0.0;


            const auto result =
                generate_candidate_poses(
                    *ligand,
                    ligand_matching_areas,
                    ligand_triplets,
                    filtered_targets,
                    protein_grid,
                    ligand_diameter,
                    config.triplet_bin_size,
                    config.operational_clash_threshold,
                    config.clash_thresholds
                );


            if (clearance_threshold) {
                std::cout << *clearance_threshold;
            } else {
                std::cout << "disabled";
            }

            std::cout
                << '\t'
                << filtered_targets.size()
                << '\t'

                << result.protein_triplet_records
                << '\t'

                << result.total_candidate_matches
                << '\t'

                << result.transforms_generated
                << '\t'

                << result
                    .poses_below_clash_threshold[
                        clash_5_index
                    ]
                << '\t'

                << result
                    .best_rmsd_below_clash_threshold[
                        clash_5_index
                    ]
                << '\t'

                << result
                    .best_rmsd_below_clash_threshold[
                        clash_10_index
                    ]
                << '\t'

                << result.timings
                    .protein_triplet_index_seconds
                << '\t'

                << result.generation_seconds;
            if (compare_clearance) {
                std::cout << '\t' << pose_targets.size() - filtered_targets.size()
                          << '\t' << protein_prep_seconds
                          << '\t' << clearance_seconds
                          << '\t' << ligand_prep_seconds
                          << '\t' << filter_seconds;
            }
            std::cout << '\n';
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
