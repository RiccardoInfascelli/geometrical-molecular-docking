#include "run_config.hpp"
#include "serial_docking.hpp"

#include <exception>
#include <iomanip>
#include <iostream>


int main(int argc, char* argv[])
{
    try {
        const RunConfig config =
            parse_run_config(
                argc,
                argv
            );


        constexpr std::size_t TOP_N = 20;


        const SerialDockingResult result =
            run_serial_docking(
                config,
                TOP_N
            );


        std::cout
            << std::fixed
            << std::setprecision(6);


        // ====================================================
        // Final predictions
        // ====================================================

        std::cout
            << "=== SERIAL DOCKING ===\n"
            << "Complex: "
            << config.complex_id
            << "\n"
            << "Returned poses: "
            << result.top_poses.size()
            << "\n\n";


        std::cout
            << "rank"
            << "\tfast_score"
            << "\tweighted_interaction_score"
            << "\tsatisfied_ligand_areas"
            << "\ttotal_interactions"
            << "\toccupied_atoms"
            << '\n';


        for (
            std::size_t i = 0;
            i < result.top_poses.size();
            ++i
        ) {
            const auto& pose =
                result.top_poses[i];


            std::cout
                << (i + 1)
                << '\t'
                << pose.fast_score
                << '\t'
                << pose.weighted_interaction_score.weighted_score
                << '\t'
                << pose.weighted_interaction_score.satisfied_ligand_areas
                << '\t'
                << pose.weighted_interaction_score.total_interactions
                << '\t'
                << pose.occupied_atoms
                << '\n';
        }


        // ====================================================
        // Timings
        // ====================================================

        const auto& t =
            result.timings;


        std::cout
            << "\n=== TIMINGS ===\n"

            << "Offline target preparation: "
            << t.offline_target_preparation_seconds
            << " s\n"

            << "Ligand preparation: "
            << t.ligand_preparation_seconds
            << " s\n"

            << "Protein triplet index: "
            << t.protein_triplet_index_seconds
            << " s\n"

            << "Matching + Kabsch: "
            << t.matching_kabsch_seconds
            << " s\n"

            << "Grid + clash: "
            << t.grid_clash_seconds
            << " s\n"

            << "FAST scoring: "
            << t.fast_scoring_seconds
            << " s\n"

            << "WeightedInteraction scoring: "
            << t.weighted_interaction_scoring_seconds
            << " s\n"

            << "Online total: "
            << t.online_total_seconds
            << " s\n"

            << "Total: "
            << t.total_seconds
            << " s\n";


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
