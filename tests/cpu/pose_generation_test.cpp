#include "clash.hpp"
#include "pose_generation.hpp"
#include "pose_targets.hpp"

#include <interaction/elements.hpp>

#include <GraphMol/Atom.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/RWMol.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct ScopedEnvironment
{
    std::string name;
    std::string previous;
    bool had_previous;

    ScopedEnvironment(const char* variable, const char* value)
        : name(variable), had_previous(std::getenv(variable) != nullptr)
    {
        if (had_previous) previous = std::getenv(variable);
        require((value ? setenv(variable, value, 1) : unsetenv(variable)) == 0,
                "Cannot configure test environment");
    }

    ~ScopedEnvironment()
    {
        if (had_previous) setenv(name.c_str(), previous.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

void test_stage_profile_output()
{
    for (const char* value : std::array<const char*, 5>{nullptr, "0", "1", "true", ""}) {
        ScopedEnvironment selector("DOCKING_PROFILE_STAGES", value);
        require(stage_profiling_enabled() == (value && std::string_view(value) == "1"),
                "Only DOCKING_PROFILE_STAGES=1 may enable profiling");
    }
    std::ostringstream output;
    output.setf(std::ios_base::scientific, std::ios_base::floatfield);
    output.precision(3);
    const auto flags = output.flags();
    const auto precision = output.precision();
    write_stage_profile(output, "cpu", 7, {{"kabsch_fit", 1.25, 12, "replay_cpu_batch"}});
    require(output.str() ==
        "PROFILE_STAGE,backend,record_index,phase,milliseconds,elements,scope\n"
        "PROFILE_STAGE,cpu,7,kabsch_fit,1.250000,12,replay_cpu_batch\n",
        "Stage profiling must use the declared diagnostic CSV schema and milliseconds");
    require(output.flags() == flags && output.precision() == precision,
            "Profiling output must preserve the caller's formatting");
}

RDKit::RWMol make_ligand(const std::vector<RDGeom::Point3D>& positions)
{
    RDKit::RWMol ligand;
    auto conformer = std::make_unique<RDKit::Conformer>(
        static_cast<unsigned int>(positions.size()));
    for (unsigned int i = 0; i < positions.size(); ++i) {
        ligand.addAtom(new RDKit::Atom(6), true, true);
        conformer->setAtomPos(i, positions[i]);
    }
    ligand.addConformer(conformer.release(), true);
    return ligand;
}

void require_same_targets(const std::vector<PoseTarget>& expected,
                          const std::vector<PoseTarget>& actual)
{
    require(expected.size() == actual.size(), "Target pruning changed the expected target count");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(expected[i].source_area_index == actual[i].source_area_index &&
                expected[i].key == actual[i].key &&
                expected[i].position.x == actual[i].position.x &&
                expected[i].position.y == actual[i].position.y &&
                expected[i].position.z == actual[i].position.z,
                "Target pruning must preserve positions, identities and order");
    }
}

void test_target_clearance_pruning()
{
    const auto protein = make_ligand({{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}});
    const auto carbon = interaction::parse_chemical_element("C");
    require(carbon.has_value(), "Synthetic protein requires a known carbon element");
    const double radius = interaction::get(*carbon).vdw_radius;
    require(std::abs(radius - 1.7) < 1e-6,
            "Synthetic clearance thresholds assume a carbon VDW radius of 1.7 A");

    // The target at x=8 is limited by the second atom, not the first.
    const std::vector<PoseTarget> targets{
        {17, {4.0, 0.0, 0.0}, InteractionKey::Hydrophobic},
        {4, {0.0, 0.0, 0.0}, InteractionKey::HydrogenDonor},
        {17, {8.0, 0.0, 0.0}, InteractionKey::HydrogenAcceptor},
        {9, {2.75, 0.0, 0.0}, InteractionKey::Metal},
        {3, {1.0, 0.0, 0.0}, InteractionKey::Hydrophobic},
        {2, {3.0, 0.0, 0.0}, InteractionKey::HydrogenDonor}
    };

    RDKit::RWMol protein_without_conformer;
    auto disabled = targets;
    const auto* original_data = disabled.data();
    require(prune_pose_targets(disabled, protein_without_conformer) == 0 &&
            disabled.data() == original_data,
            "Disabled pruning must not inspect RDKit coordinates or replace the target buffer");
    require_same_targets(targets, disabled);

    auto zero = targets;
    require(prune_pose_targets(zero, protein, 0.0) == 2,
            "A zero threshold must remove targets inside the protein VDW surface");
    require_same_targets({targets[0], targets[2], targets[3], targets[5]}, zero);

    auto low = targets;
    require(prune_pose_targets(low, protein, 0.75) == 3,
            "The 0.75 A threshold must use distance to the nearest VDW surface");
    require_same_targets({targets[0], targets[3], targets[5]}, low);

    auto high = targets;
    require(prune_pose_targets(high, protein, 1.25) == 4,
            "The 1.25 A threshold must remove the target with 1.05 A clearance");
    require_same_targets({targets[0], targets[5]}, high);

    auto boundary = targets;
    require(prune_pose_targets(boundary, protein, 2.0 - radius) == 2,
            "Clearance equal to the threshold must be retained");
    require_same_targets({targets[0], targets[2], targets[3], targets[5]}, boundary);

    auto all_pruned = targets;
    require(prune_pose_targets(all_pruned, protein, 100.0) == targets.size() &&
            all_pruned.empty(), "All targets may be pruned without an implicit fallback");
    std::vector<PoseTarget> empty;
    require(prune_pose_targets(empty, protein, 1.25) == 0 && empty.empty(),
            "Pruning an empty target set must remain valid");

    for (double threshold : {-1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        auto unchanged = targets;
        bool rejected = false;
        try {
            prune_pose_targets(unchanged, protein, threshold);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "Invalid clearance thresholds must be rejected");
        require_same_targets(targets, unchanged);
    }
}

std::vector<interaction::point3D> cache_coordinates(const RDKit::ROMol& ligand)
{
    std::vector<interaction::point3D> points;
    const auto& conformer = ligand.getConformer();
    for (unsigned int i = 0; i < ligand.getNumAtoms(); ++i) {
        const auto& position = conformer.getAtomPos(i);
        points.push_back({
            static_cast<interaction::coordinate_type>(position.x),
            static_cast<interaction::coordinate_type>(position.y),
            static_cast<interaction::coordinate_type>(position.z)
        });
    }
    return points;
}

void require_cached_clash_matches(
    const RDKit::ROMol& ligand,
    const RigidTransform& transform,
    const ProteinGrid& grid)
{
    const auto original = score_fast_grid_clash(ligand, transform, grid);
    const auto cached = score_fast_grid_clash_cached(cache_coordinates(ligand), transform, grid);
    require(original.total_atoms == cached.total_atoms &&
            original.occupied_atoms == cached.occupied_atoms &&
            original.fraction == cached.fraction,
            "Cached clash must preserve occupancy and the denominator exactly");
}

void test_fast_clash()
{
    ProteinGrid grid;
    grid.voxel_size = 1.0;
    grid.nx = grid.ny = grid.nz = 4;
    grid.voxels.resize(64);
    grid.voxels[grid.index(1, 1, 1)].occupied = true;
    grid.voxels[grid.index(2, 1, 1)].occupied = true;
    const auto ligand = make_ligand({
        {1.25, 1.25, 1.25}, {0.25, 0.25, 0.25}, {5.0, 5.0, 5.0}
    });

    // Same occupied/free/outside-grid case as the CUDA clash test.
    RigidTransform transform;
    for (double translation : {0.0, 1.0}) {
        transform.translation[0] = translation;
        const auto score = score_fast_grid_clash(ligand, transform, grid);
        require(score.total_atoms == 3 && score.occupied_atoms == 1 &&
                std::abs(score.fraction - 1.0 / 3.0) < 1e-12,
                "Fast clash must retain outside-grid atoms in the denominator");
        require_cached_clash_matches(ligand, transform, grid);
    }

    transform.rotation = {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    transform.translation = {3.0, 0.0, 0.0};
    require_cached_clash_matches(ligand, transform, grid);

    grid.voxels[grid.index(0, 1, 1)].occupied = true;
    const auto boundary_ligand = make_ligand({
        {-0.25, 1.25, 1.25}, {4.0, 1.25, 1.25}, {0.0, 1.25, 1.25}
    });
    const auto boundary = score_fast_grid_clash(boundary_ligand, RigidTransform{}, grid);
    require(boundary.total_atoms == 3 && boundary.occupied_atoms == 1 &&
            std::abs(boundary.fraction - 1.0 / 3.0) < 1e-12,
            "Fast clash grid bounds or floor conversion changed");
    require_cached_clash_matches(boundary_ligand, RigidTransform{}, grid);

    // Values on either side of a boundary narrow onto that boundary in float.
    const auto precision_ligand = make_ligand({
        {1.0 - 1e-9, 1.25, 1.25}, {4.0 - 1e-8, 1.25, 1.25},
        {2.0 + 1e-8, 1.25, 1.25}
    });
    require(score_fast_grid_clash(precision_ligand, RigidTransform{}, grid).occupied_atoms == 2,
            "Clash must narrow coordinates before voxel lookup");
    require_cached_clash_matches(precision_ligand, RigidTransform{}, grid);

    const auto empty = make_ligand({});
    const auto empty_score = score_fast_grid_clash(empty, RigidTransform{}, grid);
    require(empty_score.total_atoms == 0 && empty_score.occupied_atoms == 0 &&
            empty_score.fraction == 0.0, "Empty ligand clash must be zero");
    require_cached_clash_matches(empty, RigidTransform{}, grid);
}

void require_same_search(
    const PoseGenerationResult& full,
    const PoseGenerationResult& alternative,
    const std::vector<MatchingArea>& areas,
    const ProteinGrid& grid,
    bool screening_only)
{
    require(full.protein_triplet_records == alternative.protein_triplet_records &&
            full.protein_hash_keys == alternative.protein_hash_keys &&
            full.ligand_triplets_with_match == alternative.ligand_triplets_with_match &&
            full.total_candidate_matches == alternative.total_candidate_matches &&
            full.largest_bucket == alternative.largest_bucket &&
            full.transforms_generated == alternative.transforms_generated &&
            full.transforms_valid == alternative.transforms_valid &&
            full.poses_below_clash_threshold == alternative.poses_below_clash_threshold &&
            full.poses_zero_clash == alternative.poses_zero_clash &&
            full.poses_one_clash == alternative.poses_one_clash &&
            full.poses_two_clashes == alternative.poses_two_clashes &&
            full.poses_three_to_five_clashes == alternative.poses_three_to_five_clashes &&
            full.poses_over_five_clashes == alternative.poses_over_five_clashes &&
            full.surviving_poses.size() == alternative.surviving_poses.size(),
            "CPU modes must preserve candidates, clash diagnostics and survivors");

    for (std::size_t i = 0; i < full.surviving_poses.size(); ++i) {
        const auto& expected = full.surviving_poses[i];
        const auto& actual = alternative.surviving_poses[i];
        require(expected.transform.rotation == actual.transform.rotation &&
                expected.transform.translation == actual.transform.translation &&
                expected.occupied_atoms == actual.occupied_atoms &&
                expected.clash_fraction == actual.clash_fraction,
                "CPU modes must preserve transforms and survivor order");
        const auto expected_fast = score_matching_areas_weighted(
            transform_matching_areas(areas, expected.transform), grid);
        const auto actual_fast = score_matching_areas_weighted(
            transform_matching_areas(areas, actual.transform), grid);
        require(expected_fast.weighted_score == actual_fast.weighted_score,
                "CPU modes must preserve independently computed Fast scores");
        require(screening_only ? actual.compatible_areas == 0 && std::isnan(actual.rmsd)
                               : expected.compatible_areas == actual.compatible_areas &&
                                 expected.rmsd == actual.rmsd,
                "Unexpected survivor diagnostics for the selected CPU mode");
    }

    if (screening_only) {
        require(alternative.poses_with_grid_support == 0 && alternative.best_grid_score == 0 &&
                alternative.best_grid_score_zero_clash == 0 &&
                alternative.zero_clash_grid_histogram == std::vector<std::size_t>{1, 0, 0, 0} &&
                std::isinf(alternative.best_zero_clash_rmsd) &&
                alternative.best_zero_clash_rmsd > 0.0 &&
                alternative.timings.oracle_rmsd_seconds == 0.0,
                "Screening-only must leave omitted diagnostics unavailable and RMSD time zero");
        for (double rmsd : alternative.best_rmsd_below_clash_threshold) {
            require(std::isinf(rmsd) && rmsd > 0.0,
                    "Screening-only RMSD aggregates must retain their infinity sentinel");
        }
    } else {
        require(full.poses_with_grid_support == alternative.poses_with_grid_support &&
                full.best_grid_score == alternative.best_grid_score &&
                full.best_grid_score_zero_clash == alternative.best_grid_score_zero_clash &&
                full.zero_clash_grid_histogram == alternative.zero_clash_grid_histogram &&
                full.best_zero_clash_rmsd == alternative.best_zero_clash_rmsd &&
                full.best_rmsd_below_clash_threshold == alternative.best_rmsd_below_clash_threshold,
                "Cached clash must preserve full grid and RMSD diagnostics");
    }
}

void require_profile_matches(
    const CpuPoseStageProfile& profile,
    const PoseGenerationResult& generated,
    const std::vector<MatchingArea>& areas,
    const ProteinGrid& grid)
{
    require(profile.candidate_matches == generated.total_candidate_matches &&
            profile.transforms_generated == generated.transforms_generated &&
            profile.transforms_valid == generated.transforms_valid &&
            profile.survivors == generated.surviving_poses.size(),
            "Batch replay must preserve candidate, fit and survivor counts");
    bool has_score = false;
    double best_score = 0.0;
    for (const auto& pose : generated.surviving_poses) {
        const auto scored = score_matching_areas_weighted(
            transform_matching_areas(areas, pose.transform), grid);
        if (!has_score || scored.weighted_score > best_score) {
            best_score = scored.weighted_score;
            has_score = true;
        }
    }
    require(profile.has_fast_score == has_score && profile.best_fast_score == best_score,
            "Batch replay must score the same survivor pool");
    const std::array<std::string_view, 9> phases{
        "query_preparation", "lookup", "matching_expansion", "kabsch_fit",
        "atom_preparation", "atom_transform_clash", "survivor_selection",
        "fast_scoring", "replay_total"};
    require(profile.stages.size() == phases.size(), "Missing CPU profiling batch");
    for (std::size_t i = 0; i < phases.size(); ++i) {
        require(profile.stages[i].phase == phases[i] &&
                profile.stages[i].scope == "replay_cpu_batch" &&
                std::isfinite(profile.stages[i].milliseconds) &&
                profile.stages[i].milliseconds >= 0.0,
                "CPU stage names, scopes and units must stay explicit");
    }
    require(profile.stages[3].elements == generated.transforms_generated &&
            profile.stages[5].elements == generated.transforms_valid &&
            profile.stages[7].elements == generated.surviving_poses.size(),
            "Profiling batches must report the actual element counts");
}

void test_pose_diagnostics()
{
    ScopedEnvironment original_clash("DOCKING_CPU_CLASH_METHOD", nullptr);
    // The common API must keep diagnostics even if a screening application opts out.
    ScopedEnvironment screening_environment("DOCKING_CPU_SCREENING_ONLY", "1");
    std::vector<RDGeom::Point3D> positions;
    for (int i = 0; i < 20; ++i) {
        positions.emplace_back(0.25, 0.25, i + 0.25);
    }
    const auto ligand = make_ligand(positions);
    const std::vector<MatchingArea> areas{
        {0, {0.25, 0.25, 0.25}, InteractionKey::Hydrophobic},
        {1, {1.25, 0.25, 0.25}, InteractionKey::HydrogenDonor},
        {2, {0.25, 1.25, 0.25}, InteractionKey::HydrogenAcceptor}
    };
    const auto triplets = generate_triplets(areas, 2.0);
    require(triplets.size() == 1, "Synthetic fixture must have one ligand triplet");

    ProteinGrid grid;
    grid.voxel_size = 1.0;
    grid.nx = 20;
    grid.ny = 3;
    grid.nz = 20;
    grid.voxels.resize(1200);
    for (auto& voxel : grid.voxels) {
        for (const auto& area : areas) {
            add_interaction(voxel.interactions, area.key);
        }
    }
    // Translations 0/4/8/12/16 give 0/1/2/3/6 occupied atoms out of 20.
    for (int i = 0; i < 6; ++i) {
        if (i < 1) grid.voxels[grid.index(4, 0, i)].occupied = true;
        if (i < 2) grid.voxels[grid.index(8, 0, i)].occupied = true;
        if (i < 3) grid.voxels[grid.index(12, 0, i)].occupied = true;
        grid.voxels[grid.index(16, 0, i)].occupied = true;
    }

    std::vector<PoseTarget> targets;
    FlatTripletIndex index;
    index.keys.push_back(make_triplet_key(canonicalize_triplet(triplets[0], 0.5)));
    // Deliberately unsorted poses: surviving translations must remain 4, then 0.
    for (int translation : {8, 4, 12, 0, 16}) {
        const auto first = static_cast<std::uint32_t>(targets.size());
        for (const auto& area : areas) {
            auto point = area.centroid;
            point.x += static_cast<interaction::coordinate_type>(translation);
            targets.push_back({targets.size(), point, area.key});
        }
        index.records.push_back({{first, first + 1, first + 2}});
    }
    index.offsets = {0, index.records.size()};

    const auto strict = generate_candidate_poses_from_index(
        ligand, areas, triplets, targets, grid, index, 0.5, 0.05,
        {0.0, 0.025, 0.05, 0.10});
    require(strict.poses_below_clash_threshold == std::array<std::size_t, 4>{1, 1, 2, 3},
            "Diagnostics must include the 10% pose rejected by the operational filter");
    require(strict.poses_three_to_five_clashes == 0 && strict.poses_over_five_clashes == 2,
            "Historical early-rejection diagnostic count changed");

    const auto wide = generate_candidate_poses_from_index(
        ligand, areas, triplets, targets, grid, index, 0.5, 0.05,
        {0.0, 0.05, 0.10, 1.0});
    require(wide.poses_below_clash_threshold == std::array<std::size_t, 4>{1, 2, 3, 5},
            "Clash threshold diagnostics changed");
    require(wide.poses_three_to_five_clashes == 1 && wide.poses_over_five_clashes == 1,
            "Absolute clash diagnostic buckets changed");

    for (const auto* result : {&strict, &wide}) {
        require(result->protein_triplet_records == 5 && result->protein_hash_keys == 1 &&
                result->ligand_triplets_with_match == 1 && result->total_candidate_matches == 5 &&
                result->largest_bucket == 5 && result->transforms_generated == 5 &&
                result->transforms_valid == 5, "Candidate or transform counts changed");
        require(result->poses_with_grid_support == 5 && result->best_grid_score == 3 &&
                result->poses_zero_clash == 1 && result->poses_one_clash == 1 &&
                result->poses_two_clashes == 1 && result->best_grid_score_zero_clash == 3 &&
                result->zero_clash_grid_histogram == std::vector<std::size_t>{0, 0, 0, 1} &&
                std::abs(result->best_zero_clash_rmsd) < 1e-9,
                "Grid support or zero-clash diagnostics changed");
        for (double rmsd : result->best_rmsd_below_clash_threshold) {
            require(std::abs(rmsd) < 1e-9, "Best diagnostic RMSD changed");
        }
        require(result->surviving_poses.size() == 2, "Operational survivor count changed");
        const auto& first = result->surviving_poses[0];
        const auto& second = result->surviving_poses[1];
        require(std::abs(first.transform.translation[0] - 4.0) < 1e-9 &&
                std::abs(second.transform.translation[0]) < 1e-9 &&
                first.occupied_atoms == 1 && second.occupied_atoms == 0 &&
                first.clash_fraction == 0.05 && second.clash_fraction == 0.0 &&
                first.compatible_areas == 2 && second.compatible_areas == 3 &&
                std::abs(first.rmsd - 4.0) < 1e-9 && std::abs(second.rmsd) < 1e-9,
                "Survivor order or pose diagnostic values changed");
    }

    for (const char* method : {"original", "cached"}) {
        ScopedEnvironment clash_method("DOCKING_CPU_CLASH_METHOD", method);
        const auto full = generate_candidate_poses_from_index(
            ligand, areas, triplets, targets, grid, index, 0.5, 0.05,
            {0.0, 0.025, 0.05, 0.10});
        require_same_search(strict, full, areas, grid, false);
        const auto screening = generate_candidate_poses_from_index(
            ligand, areas, triplets, targets, grid, index, 0.5, 0.05,
            {0.0, 0.025, 0.05, 0.10}, true);
        require_same_search(strict, screening, areas, grid, true);

        const auto profile = profile_cpu_pose_stages(
            ligand, areas, triplets, targets, grid, index, 0.5, 0.05,
            {0.0, 0.025, 0.05, 0.10});
        require_profile_matches(profile, screening, areas, grid);

        // Duplicates remain separate candidates, and the diagnostic limit is
        // still applied before a deliberately looser operational threshold.
        const std::vector<Triplet> repeated_triplets{triplets[0], triplets[0]};
        const auto gated = generate_candidate_poses_from_index(
            ligand, areas, repeated_triplets, targets, grid, index, 0.5, 1.0,
            {0.0, 0.025, 0.05, 0.05}, true);
        const auto gated_profile = profile_cpu_pose_stages(
            ligand, areas, repeated_triplets, targets, grid, index, 0.5, 1.0,
            {0.0, 0.025, 0.05, 0.05});
        require_profile_matches(gated_profile, gated, areas, grid);
        require(gated_profile.transforms_generated == 10 && gated_profile.survivors == 4,
                "Batch profiling must preserve duplicate candidates and early filtering");

        const auto empty = generate_candidate_poses_from_index(
            ligand, areas, {}, targets, grid, index, 0.5, 0.05,
            {0.0, 0.025, 0.05, 0.10}, true);
        require_profile_matches(profile_cpu_pose_stages(
            ligand, areas, {}, targets, grid, index, 0.5, 0.05,
            {0.0, 0.025, 0.05, 0.10}), empty, areas, grid);

        const auto after_profile = generate_candidate_poses_from_index(
            ligand, areas, triplets, targets, grid, index, 0.5, 0.05,
            {0.0, 0.025, 0.05, 0.10}, true);
        require_same_search(screening, after_profile, areas, grid, true);
    }
}

void test_oversized_ligand_shared_index()
{
    constexpr double cutoff = 1.5;
    constexpr double bin_size = 0.5;
    const auto ligand = make_ligand({
        {0.25, 0.25, 0.25}, {1.25, 0.25, 0.25},
        {0.25, 1.25, 0.25}, {10.25, 0.25, 0.25}
    });
    const std::vector<MatchingArea> areas{
        {0, {0.25, 0.25, 0.25}, InteractionKey::Hydrophobic},
        {1, {1.25, 0.25, 0.25}, InteractionKey::HydrogenDonor},
        {2, {0.25, 1.25, 0.25}, InteractionKey::HydrogenAcceptor},
        {3, {10.25, 0.25, 0.25}, InteractionKey::Metal}
    };
    std::vector<interaction::interaction_area> physical_areas(areas.size());
    for (std::size_t i = 0; i < areas.size(); ++i)
        physical_areas[i].centroid = areas[i].centroid;
    const double diameter = compute_ligand_diameter(physical_areas);
    require(diameter > cutoff, "Synthetic ligand must exceed the shared cutoff");
    const auto triplets = generate_triplets(areas, diameter);
    require(triplets.size() > 1, "Long ligand triplets must remain in the query population");

    std::vector<PoseTarget> targets;
    for (std::size_t i = 0; i < 3; ++i)
        targets.push_back({i, areas[i].centroid, areas[i].key});
    const auto hash = build_protein_triplet_hash_table(targets, cutoff, bin_size);
    const auto shared_index = flatten_protein_triplet_index(hash);
    require(shared_index.records.size() == 1, "Shared fixture must contain one protein triplet");

    auto disabled_targets = targets;
    RDKit::RWMol protein_without_conformer;
    require(prune_pose_targets(disabled_targets, protein_without_conformer, std::nullopt) == 0,
            "Disabled pruning must preserve every target before building the shared index");
    require_same_targets(targets, disabled_targets);
    const auto disabled_hash = build_protein_triplet_hash_table(disabled_targets, cutoff, bin_size);
    const auto disabled_index = flatten_protein_triplet_index(disabled_hash);
    require(shared_index.keys == disabled_index.keys &&
            shared_index.offsets == disabled_index.offsets &&
            shared_index.records.size() == disabled_index.records.size(),
            "Disabled pruning must preserve the shared index keys and buckets");
    for (std::size_t i = 0; i < shared_index.records.size(); ++i) {
        require(shared_index.records[i].indices == disabled_index.records[i].indices,
                "Disabled pruning must preserve the shared index record order");
    }

    ProteinGrid grid;
    grid.voxel_size = 1.0;
    grid.nx = 12;
    grid.ny = grid.nz = 3;
    grid.voxels.resize(108);
    for (auto& voxel : grid.voxels)
        for (const auto& area : areas) add_interaction(voxel.interactions, area.key);

    // The same cutoff-limited index is borrowed by both the search and its
    // profiling replay; the longer ligand still contributes a matching pose.
    for (const char* method : {"original", "cached"}) {
        ScopedEnvironment clash_method("DOCKING_CPU_CLASH_METHOD", method);
        const auto generated = generate_candidate_poses_from_index(
            ligand, areas, triplets, targets, grid, shared_index, bin_size, 0.05,
            {0.0, 0.025, 0.05, 0.10}, true);
        require(generated.protein_triplet_records == shared_index.records.size() &&
                generated.total_candidate_matches == 1 && generated.transforms_generated == 1 &&
                generated.surviving_poses.size() == 1,
                "Oversized ligand must use the supplied shared index without being rejected");
        require_profile_matches(profile_cpu_pose_stages(
            ligand, areas, triplets, targets, grid, shared_index, bin_size, 0.05,
            {0.0, 0.025, 0.05, 0.10}), generated, areas, grid);
        require(validate_flat_triplet_index(hash, shared_index),
                "Search and profiling must leave the shared index unchanged");

        const auto baseline_full = generate_candidate_poses_from_index(
            ligand, areas, triplets, targets, grid, shared_index, bin_size, 0.05,
            {0.0, 0.025, 0.05, 0.10});
        const auto disabled_full = generate_candidate_poses_from_index(
            ligand, areas, triplets, disabled_targets, grid, disabled_index, bin_size, 0.05,
            {0.0, 0.025, 0.05, 0.10});
        require_same_search(baseline_full, disabled_full, areas, grid, false);
        require(baseline_full.surviving_poses.size() == 1,
                "Disabled pruning regression requires a unique top-1 pose");
        const auto& baseline_top = baseline_full.surviving_poses.front();
        const auto& disabled_top = disabled_full.surviving_poses.front();
        require(baseline_top.transform.rotation == disabled_top.transform.rotation &&
                baseline_top.transform.translation == disabled_top.transform.translation &&
                score_matching_areas_weighted(
                    transform_matching_areas(areas, baseline_top.transform), grid).weighted_score ==
                score_matching_areas_weighted(
                    transform_matching_areas(areas, disabled_top.transform), grid).weighted_score,
                "Disabled pruning must preserve the top-1 transform and Fast score");
    }
}

} // namespace

int main()
{
    try {
        test_stage_profile_output();
        test_target_clearance_pruning();
        test_fast_clash();
        test_pose_diagnostics();
        test_oversized_ligand_shared_index();
        std::cout << "CPU pose generation and clash tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
