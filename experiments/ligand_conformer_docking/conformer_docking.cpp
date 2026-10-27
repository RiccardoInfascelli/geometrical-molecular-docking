#include "conformer_docking.hpp"
#include "gpu_docking_output.hpp"
#include <chrono>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
}

ConformerDockingResult dock_conformer(
    std::size_t conformer_id, RDKit::RWMol& ligand, const RDKit::ROMol& reference,
    const PreparedGpuTarget& target, const RescoringTarget& scoring,
    const RunConfig& config, std::optional<std::size_t> fast_prefilter) {
    validate_gpu_docking_reference(ligand, reference);
    ConformerDockingResult result;
    auto start = Clock::now();
    PreparedLigand prepared;
    prepare_ligand(ligand, prepared);
    result.preparation_s = seconds(start);
    result.diameter = prepared.diameter;
    result.triplets = prepared.triplets.size();

    start = Clock::now();
    const auto search = run_gpu_ligand_search(ligand, prepared, target, config,
                                            nullptr, nullptr, nullptr, true);
    result.search_s = seconds(start); // Completed wrapper, including queries and copies.
    result.query_s = search.query_preparation_seconds;
    result.matches = search.candidate_matches;
    result.alignments = search.alignments;
    result.valid_transforms = search.pipeline.valid_transform_count;
    result.survivors = search.pipeline.clash_survivor_count;
    if (result.survivors != search.pipeline.poses.size())
        throw std::runtime_error("CUDA survivor count differs from the downloaded pool");

    const auto candidates = make_rescoring_pose_pool(search.pipeline.poses);
    start = Clock::now();
    const auto rescored = rescore_poses(candidates, ligand, prepared, target.protein.grid,
        scoring.interaction_areas, scoring.vinardo, fast_prefilter, true);
    result.rescoring_s = seconds(start);
    result.prefilter_s = rescored.selection_seconds;
    result.weighted_interaction_s = rescored.weighted_interaction_scoring_seconds;
    result.vinardo_s = rescored.vinardo_scoring_seconds;
    result.poses.reserve(candidates.size());
    start = Clock::now();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double rmsd = gpu_pose_heavy_atom_rmsd(ligand, search.pipeline.poses[i].transform, reference);
        result.poses.push_back({conformer_id, i, {i, candidates[i], {}}, rmsd,
                               search.pipeline.poses[i].alignment_index});
        if (!result.oracle_all_survivors || rmsd < *result.oracle_all_survivors)
            result.oracle_all_survivors = rmsd;
    }
    for (const auto& row : rescored.poses) {
        result.poses.at(row.pose_id).scored = row;
        result.rescored_ids.push_back(row.pose_id);
        const double rmsd = result.poses.at(row.pose_id).rmsd;
        if (!result.oracle_after_prefilter || rmsd < *result.oracle_after_prefilter)
            result.oracle_after_prefilter = rmsd;
    }
    result.rmsd_s = seconds(start);
    return result;
}
