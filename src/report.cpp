#include "report.hpp"

#include "diagnostics.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>


namespace
{

double best_rmsd_in_top(
    const std::vector<RankedPoseDiagnostic>& poses,
    std::size_t top_n)
{
    if (poses.empty()) {
        return std::numeric_limits<double>::infinity();
    }


    const std::size_t count =
        std::min(
            top_n,
            poses.size()
        );


    double best =
        std::numeric_limits<double>::infinity();


    for (
        std::size_t i = 0;
        i < count;
        ++i
    ) {
        best =
            std::min(
                best,
                poses[i].rmsd
            );
    }


    return best;
}

void print_summary(
    const RunResult& result)
{
    const double top500_best =
        best_rmsd_in_top(
            result.pose_ranking.weighted_interaction_candidates,
            result.pose_ranking.weighted_interaction_candidates.size()
        );

    const double top5_best =
        best_rmsd_in_top(
            result.pose_ranking.weighted_interaction_ranking,
            5
        );

    const double top10_best =
        best_rmsd_in_top(
            result.pose_ranking.weighted_interaction_ranking,
            10
        );


    std::cout
        << result.complex_id

        << " | matching "
        << result.pocket_matching_areas

        << " | targets "
        << result.pose_targets

        << " | candidates "
        << result.pose_generation.transforms_generated

        << " | survivors "
        << result.pose_ranking.input_pose_count

        << " | top500_best "
        << top500_best

        << " | top5_best "
        << top5_best

        << " | top10_best "
        << top10_best

        << " | offline "
        << result.timings
               .offline_target_preparation_seconds
        << " s"

        << " | ligand "
        << result.timings
               .online_ligand_preparation_seconds
        << " s"

        << " | search "
        << result.timings
               .online_pose_search_seconds
        << " s"

        << " | scoring "
        << result.timings
               .online_scoring_seconds
        << " s"

        << " | online "
        << result.timings
               .online_total_seconds
        << " s"

        << " | total "
        << result.timings.total_seconds
        << " s"

        << '\n';
}


void print_debug(
    const RunResult& result,
    const RunConfig& config)
{
    std::cout
        << "=== Molecules ===\n"

        << "Protein atoms: "
        << result.protein_atoms
        << '\n'

        << "Pocket atoms: "
        << result.pocket_atoms
        << '\n'

        << "Ligand atoms: "
        << result.ligand_atoms
        << '\n';


    // ========================================================
    // Interaction areas
    // ========================================================

    std::cout
        << "\n=== Interaction areas ===\n"

        << "Protein interaction areas: "
        << result.protein_areas
        << '\n'

        << "Pocket protein areas: "
        << result.pocket_protein_areas
        << '\n'

        << "Ligand interaction areas: "
        << result.ligand_areas
        << '\n'

        << "Protein matching areas: "
        << result.protein_matching_areas
        << '\n'

        << "Ligand matching areas: "
        << result.ligand_matching_areas
        << '\n';


    // ========================================================
    // Ligand geometry
    // ========================================================

    std::cout
        << "\n=== Ligand geometry ===\n"

        << "Ligand diameter: "
        << result.ligand_diameter
        << " A\n"

        << "Ligand triplets: "
        << result.ligand_triplets
        << '\n';


    // ========================================================
    // Protein grid
    // ========================================================

    std::cout
        << "\n=== Protein grid ===\n"

        << "Pocket size: "
        << result.pocket_size_x
        << " x "
        << result.pocket_size_y
        << " x "
        << result.pocket_size_z
        << " A\n"

        << "Grid dimensions: "
        << result.grid_nx
        << " x "
        << result.grid_ny
        << " x "
        << result.grid_nz
        << '\n'

        << "Total voxels: "
        << result.total_voxels
        << '\n'

        << "Occupied voxels: "
        << result.occupied_voxels
        << '\n'

        << "Interaction voxels: "
        << result.interaction_voxels
        << '\n'

        << "Multi-interaction voxels: "
        << result.multi_interaction_voxels
        << '\n';


    // ========================================================
    // Crystal / grid oracle
    // ========================================================

    std::cout
        << "\n=== Crystal ligand / fast grid ===\n"

        << "Areas in grid: "
        << result.crystal_grid_in_grid
        << " / "
        << result.crystal_grid_total
        << '\n'

        << "Areas on occupied voxel: "
        << result.crystal_grid_occupied
        << '\n'

        << "Grid-compatible ligand areas: "
        << result.crystal_grid_compatible
        << " / "
        << result.crystal_grid_total
        << '\n';


    // ========================================================
    // Crystal interaction oracle
    // ========================================================

    std::cout
        << "\n=== Crystal chemical interactions ===\n"

        << "Chemical interactions: "
        << result.crystal_interactions
        << '\n'

        << "Interacting ligand areas: "
        << result.crystal_interacting_ligand_areas
        << '\n';


    std::cout
        << "\n=== Pocket-only chemical interaction check ===\n"

        << "Full protein areas: "
        << result.protein_areas
        << '\n'

        << "Pocket protein areas: "
        << result.pocket_protein_areas
        << '\n'

        << "Detected protein-area chemical interactions: "
        << result.crystal_interactions
        << '\n'

        << "Pocket chemical interactions: "
        << result.pocket_interactions
        << '\n'

        << "Ligand areas interacting with detected protein areas: "
        << result.crystal_interacting_ligand_areas
        << '\n'

        << "Pocket interacting ligand areas: "
        << result.pocket_interacting_ligand_areas
        << '\n';


    // ========================================================
    // Matching-area diagnostics
    // ========================================================

    print_matching_area_distance_diagnostic(
        result.matching_area_distance_diagnostic
    );


    // ========================================================
    // Pose targets
    // ========================================================

    std::cout
        << "\n=== Pose targets ===\n"

        << "Protein matching areas: "
        << result.protein_matching_areas
        << '\n'

        << "Matching areas in pocket: "
        << result.pocket_matching_areas
        << '\n'

        << "Pose targets generated: "
        << result.pose_targets
        << '\n'

        << "Pocket areas without target: "
        << result.pose_targets_missing
        << '\n';


    print_pose_target_crystal_diagnostic(
        result.pose_target_crystal_diagnostic
    );


    // ========================================================
    // Pose generation diagnostics
    // ========================================================

    print_pose_generation_diagnostics(
        result.pose_generation,
        result.pose_targets,
        result.ligand_triplets,
        result.ligand_matching_areas,
        result.ligand_atoms,
        config.clash_thresholds
    );


    // ========================================================
    // Existing ranking diagnostics
    // ========================================================

    print_pose_ranking_diagnostics(
        result.pose_ranking
    );


    // ========================================================
    // FAST vs WeightedInteraction-50 vs WeightedInteraction-500
    //
    // RMSD evaluates completed rankings and does not affect pose selection.
    // ========================================================

    const auto best_rmsd_in_top =
        [](
            const std::vector<RankedPoseDiagnostic>& poses,
            std::size_t top_n)
        {
            double best =
                std::numeric_limits<double>::infinity();


            const std::size_t count =
                std::min(
                    top_n,
                    poses.size()
                );


            for (
                std::size_t i = 0;
                i < count;
                ++i
            ) {
                best =
                    std::min(
                        best,
                        poses[i].rmsd
                    );
            }


            return best;
        };


    const auto print_ablation =
        [&best_rmsd_in_top](
            const char* name,
            const std::vector<
                RankedPoseDiagnostic
            >& ranking)
        {
            std::cout
                << '\n'
                << name
                << '\n'

                << "  poses: "
                << ranking.size()
                << '\n'

                << "  best RMSD top 1:  "
                << best_rmsd_in_top(
                    ranking,
                    1
                )
                << " A\n"

                << "  best RMSD top 5:  "
                << best_rmsd_in_top(
                    ranking,
                    5
                )
                << " A\n"

                << "  best RMSD top 10: "
                << best_rmsd_in_top(
                    ranking,
                    10
                )
                << " A\n"

                << "  best RMSD top 20: "
                << best_rmsd_in_top(
                    ranking,
                    20
                )
                << " A\n";
        };


    std::cout
        << "\n=== Scoring ablation ===\n";


    print_ablation(
        "FAST ONLY",
        result.pose_ranking
            .fast_weighted_ranking
    );


    print_ablation(
        "WeightedInteraction reranking / FAST TOP 50",
        result.pose_ranking
            .weighted_interaction_top50_ranking
    );


    print_ablation(
        "WeightedInteraction reranking / FAST TOP 500",
        result.pose_ranking
            .weighted_interaction_ranking
    );


    // ========================================================
    // Scoring timing comparison
    // ========================================================

    std::cout
        << "\n=== Scoring timing comparison ===\n"

        << "Fast-only scoring: "
        << result.pose_ranking
               .fast_scoring_seconds
        << " s\n"

        << "WeightedInteraction scoring first 50: "
        << result.pose_ranking
               .weighted_interaction_top50_scoring_seconds
        << " s\n"

        << "WeightedInteraction scoring first 500: "
        << result.pose_ranking
               .weighted_interaction_scoring_seconds
        << " s\n";


    // ========================================================
    // Pipeline profiling
    // ========================================================

    std::cout
        << "\n=== Pipeline profiling ===\n"

        << "OFFLINE target preparation: "
        << result.timings
               .offline_target_preparation_seconds
        << " s\n"

        << '\n'

        << "ONLINE ligand preparation: "
        << result.timings
               .online_ligand_preparation_seconds
        << " s\n"

        << "ONLINE pose search: "
        << result.timings
               .online_pose_search_seconds
        << " s\n"

        << "ONLINE scoring / ranking: "
        << result.timings
               .online_scoring_seconds
        << " s\n"

        << "ONLINE total: "
        << result.timings
               .online_total_seconds
        << " s\n"

        << '\n'

        << "Pipeline total: "
        << result.timings.total_seconds
        << " s\n";


    // ========================================================
    // Pose-search internal profiling
    // ========================================================

    std::cout
        << "\n=== Pose search breakdown ===\n"

        << "Protein triplet index: "
        << result.pose_generation.timings
               .protein_triplet_index_seconds
        << " s\n"

        << "Matching + Kabsch: "
        << result.pose_generation.timings
               .matching_kabsch_seconds
        << " s\n"

        << "Grid + clash: "
        << result.pose_generation.timings
               .grid_clash_seconds
        << " s\n"

        << "Oracle RMSD diagnostic: "
        << result.pose_generation.timings
               .oracle_rmsd_seconds
        << " s\n";
}

} // namespace


void print_report(
    const RunResult& result,
    const RunConfig& config)
{
    std::cout
        << std::setprecision(6);


    switch (config.output_mode) {

        case OutputMode::Quiet:
            return;


        case OutputMode::Summary:

            print_summary(
                result
            );

            return;


        case OutputMode::Debug:

            print_debug(
                result,
                config
            );

            return;
    }
}
