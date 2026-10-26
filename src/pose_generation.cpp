#include "pose_generation.hpp"

#include "clash.hpp"
#include "rigid_alignment.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>



namespace

{



// Atom-index RMSD to the input conformer, with no symmetry matching or refit.
// This diagnostic uses all atoms and never participates in pose selection.
double rmsd_from_transform(

    const RDKit::ROMol& ligand,

    const RigidTransform& transform)

{

    const auto& conformer =

        ligand.getConformer();



    double squared_sum = 0.0;



    for (

        unsigned int atom_index = 0;

        atom_index < ligand.getNumAtoms();

        ++atom_index

    ) {

        const auto position =

            conformer.getAtomPos(

                atom_index

            );



        const interaction::point3D original{

            static_cast<interaction::coordinate_type>(

                position.x

            ),

            static_cast<interaction::coordinate_type>(

                position.y

            ),

            static_cast<interaction::coordinate_type>(

                position.z

            )

        };



        const auto transformed =

            transform_point(

                original,

                transform

            );



        const double dx =

            static_cast<double>(

                transformed.x

            ) - position.x;



        const double dy =

            static_cast<double>(

                transformed.y

            ) - position.y;



        const double dz =

            static_cast<double>(

                transformed.z

            ) - position.z;



        squared_sum +=

            dx * dx +

            dy * dy +

            dz * dz;

    }



    return std::sqrt(

        squared_sum /

        static_cast<double>(

            ligand.getNumAtoms()

        )

    );

}



// Called after the diagnostic clash limit and optional RMSD calculation, before the
// operational survivor filter. These statistics include operational rejects.
void update_clash_diagnostics(
    PoseGenerationResult& result,
    const FastClashScore& clash,
    const GridPoseScore& grid_score,
    double rmsd,
    const std::array<double, 4>& clash_thresholds)
{
    for (std::size_t threshold_index = 0;
         threshold_index < clash_thresholds.size();
         ++threshold_index) {
        if (clash.fraction <= clash_thresholds[threshold_index]) {
            ++result.poses_below_clash_threshold[threshold_index];
            result.best_rmsd_below_clash_threshold[threshold_index] = std::min(
                result.best_rmsd_below_clash_threshold[threshold_index], rmsd);
        }
    }

    if (clash.occupied_atoms == 0) {
        ++result.poses_zero_clash;
        result.best_zero_clash_rmsd = std::min(result.best_zero_clash_rmsd, rmsd);
        result.best_grid_score_zero_clash = std::max(
            result.best_grid_score_zero_clash, grid_score.compatible_areas);
        ++result.zero_clash_grid_histogram[grid_score.compatible_areas];
    }
    else if (clash.occupied_atoms == 1) {
        ++result.poses_one_clash;
    }
    else if (clash.occupied_atoms == 2) {
        ++result.poses_two_clashes;
    }
    else if (clash.occupied_atoms <= 5) {
        ++result.poses_three_to_five_clashes;
    }
    else {
        ++result.poses_over_five_clashes;
    }
}

} // namespace



PoseGenerationResult generate_candidate_poses_from_index(

    const RDKit::ROMol& ligand,

    const std::vector<MatchingArea>& ligand_matching_areas,

    const std::vector<Triplet>& ligand_triplets,

    const std::vector<PoseTarget>& pose_targets,

    const ProteinGrid& protein_grid,

    const FlatTripletIndex& protein_triplet_index,

    double triplet_bin_size,

    double operational_clash_threshold,

    const std::array<double, 4>& clash_thresholds,

    bool screening_only)

{

    PoseGenerationResult result;

    const char* clash_method = std::getenv("DOCKING_CPU_CLASH_METHOD");
    const bool use_cached_clash = clash_method &&
        std::strcmp(clash_method, "cached") == 0;
    std::vector<interaction::point3D> ligand_points;



    result.zero_clash_grid_histogram.assign(

        ligand_matching_areas.size() + 1,

        0

    );



    using Clock =

        std::chrono::steady_clock;



    // ========================================================
    // Protein triplet index statistics
    //
    // The caller supplies an already built index.
    // ========================================================



    result.protein_triplet_records =
        protein_triplet_index.records.size();

    result.protein_hash_keys =
        protein_triplet_index.keys.size();



    // ========================================================
    // Candidate pose generation
    // ========================================================



    double grid_clash_seconds = 0.0;

    double oracle_rmsd_seconds = 0.0;



    const auto generation_start =

        Clock::now();

    // Keep the original coordinate narrowing, once per search instead of per pose.
    // Preparation is included in generation time and its matching/Kabsch residual.
    if (use_cached_clash) {
        const auto& conformer = ligand.getConformer();
        ligand_points.reserve(ligand.getNumAtoms());
        for (unsigned int i = 0; i < ligand.getNumAtoms(); ++i) {
            const auto position = conformer.getAtomPos(i);
            ligand_points.push_back({
                static_cast<interaction::coordinate_type>(position.x),
                static_cast<interaction::coordinate_type>(position.y),
                static_cast<interaction::coordinate_type>(position.z)
            });
        }
    }



    for (const auto& ligand_triplet :

         ligand_triplets) {

        const auto canonical =

            canonicalize_triplet(

                ligand_triplet,

                triplet_bin_size

            );



        const auto buckets =
            find_neighbor_triplet_buckets_flat(
                protein_triplet_index,
                canonical
            );



        if (buckets.empty())

            continue;



        ++result.ligand_triplets_with_match;



        for (const auto& bucket :
            buckets) {

            const auto [begin, end] =
                bucket;

            const std::size_t bucket_size =
                end - begin;

            result.total_candidate_matches +=
                bucket_size;

            result.largest_bucket =
                std::max(
                    result.largest_bucket,
                    bucket_size
                );
        }



        for (const auto& bucket :
            buckets) {

            const auto [begin, end] =
                bucket;

            for (std::size_t record_index = begin;
                record_index < end;
                ++record_index) {

                const auto& protein_record =
                    protein_triplet_index.records[
                        record_index
                    ];

                const std::array<
                    interaction::point3D,
                    3
                > target_points{

                    pose_targets[

                        protein_record.indices[0]

                    ].position,

                    pose_targets[

                        protein_record.indices[1]

                    ].position,

                    pose_targets[

                        protein_record.indices[2]

                    ].position

                };



                for (

                    std::size_t order_index = 0;

                    order_index <

                        canonical.equivalent_order_count;

                    ++order_index

                ) {

                    const auto& ligand_order =

                        canonical.equivalent_orders[

                            order_index

                        ];



                    const std::array<

                        interaction::point3D,

                        3

                    > source_points{

                        ligand_matching_areas[

                            ligand_order[0]

                        ].centroid,

                        ligand_matching_areas[

                            ligand_order[1]

                        ].centroid,

                        ligand_matching_areas[

                            ligand_order[2]

                        ].centroid

                    };



                    ++result.transforms_generated;



                    RigidTransform transform;



                    if (

                        !rigid_transform_from_points(

                            source_points,

                            target_points,

                            transform

                        )

                    ) {

                        continue;

                    }



                    ++result.transforms_valid;



                    // ----------------------------------------
                    // Grid support and clash
                    // ----------------------------------------



                    const auto grid_clash_start =

                        Clock::now();



                    // Keep coordinate-buffer cleanup outside the grid/clash timer.
                    std::vector<MatchingArea> transformed_matching;
                    GridPoseScore grid_score{};
                    if (!screening_only) {
                        transformed_matching = transform_matching_areas(
                            ligand_matching_areas, transform);
                        grid_score = score_ligand_on_grid(transformed_matching, protein_grid);
                        if (grid_score.compatible_areas > 0) {
                            ++result.poses_with_grid_support;
                        }
                        result.best_grid_score = std::max(
                            result.best_grid_score, grid_score.compatible_areas);
                    }

                    const auto clash = use_cached_clash
                        ? score_fast_grid_clash_cached(ligand_points, transform, protein_grid)
                        : score_fast_grid_clash(ligand, transform, protein_grid);


                    const auto grid_clash_end =

                        Clock::now();



                    grid_clash_seconds +=

                        std::chrono::duration<double>(

                            grid_clash_end -

                            grid_clash_start

                        ).count();



                    if (

                        clash.fraction >

                        clash_thresholds.back()

                    ) {

                        ++result.poses_over_five_clashes;

                        continue;

                    }



                    // ----------------------------------------
                    // RMSD oracle only
                    // ----------------------------------------



                    double rmsd = std::numeric_limits<double>::quiet_NaN();
                    if (!screening_only) {
                        const auto rmsd_start = Clock::now();
                        rmsd = rmsd_from_transform(ligand, transform);
                        const auto rmsd_end = Clock::now();
                        oracle_rmsd_seconds += std::chrono::duration<double>(
                            rmsd_end - rmsd_start).count();
                    }


                    update_clash_diagnostics(
                        result, clash, grid_score, rmsd, clash_thresholds);



                    // ----------------------------------------
                    // Operational survivor set
                    // ----------------------------------------



                    if (

                        clash.fraction >

                        operational_clash_threshold

                    ) {

                        continue;

                    }



                    RankedPoseDiagnostic record;



                    record.compatible_areas =

                        grid_score.compatible_areas;



                    record.occupied_atoms =

                        clash.occupied_atoms;



                    record.clash_fraction =

                        clash.fraction;



                    record.rmsd =

                        rmsd;



                    record.transform =

                        transform;



                    result.surviving_poses.push_back(

                        record

                    );

                }

            }

        }

    }



    const auto generation_end =

        Clock::now();



    result.generation_seconds =

        std::chrono::duration<double>(

            generation_end -

            generation_start

        ).count();



    result.timings.grid_clash_seconds =

        grid_clash_seconds;



    result.timings.oracle_rmsd_seconds =

        oracle_rmsd_seconds;



    // Residual wall time includes lookup, fitting and loop/diagnostic bookkeeping.
    const double matching_kabsch_seconds =

        result.generation_seconds -

        grid_clash_seconds -

        oracle_rmsd_seconds;



    result.timings.matching_kabsch_seconds =

        std::max(

            0.0,

            matching_kabsch_seconds

        );



    return result;

}



CpuPoseStageProfile profile_cpu_pose_stages(
    const RDKit::ROMol& ligand,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<Triplet>& ligand_triplets,
    const std::vector<PoseTarget>& pose_targets,
    const ProteinGrid& protein_grid,
    const FlatTripletIndex& protein_triplet_index,
    double triplet_bin_size,
    double operational_clash_threshold,
    const std::array<double, 4>& clash_thresholds)
{
    using Clock = std::chrono::steady_clock;
    CpuPoseStageProfile result;
    const auto replay_start = Clock::now();
    const auto append_stage = [&](std::string_view name, Clock::time_point start,
                                  std::size_t elements) {
        const auto end = Clock::now();
        result.stages.push_back({name,
            std::chrono::duration<double, std::milli>(end - start).count(),
            elements, "replay_cpu_batch"});
    };

    std::vector<CanonicalTriplet> queries;
    queries.reserve(ligand_triplets.size());
    auto start = Clock::now();
    for (const auto& triplet : ligand_triplets) {
        queries.push_back(canonicalize_triplet(triplet, triplet_bin_size));
    }
    append_stage("query_preparation", start, queries.size());

    std::vector<std::vector<FlatTripletBucket>> query_buckets(queries.size());
    start = Clock::now();
    for (std::size_t i = 0; i < queries.size(); ++i) {
        query_buckets[i] = find_neighbor_triplet_buckets_flat(
            protein_triplet_index, queries[i]);
    }
    append_stage("lookup", start, queries.size());

    struct AlignmentPoints
    {
        std::array<interaction::point3D, 3> source;
        std::array<interaction::point3D, 3> target;
    };
    std::vector<AlignmentPoints> alignments;
    start = Clock::now();
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto& query = queries[i];
        for (const auto& bucket : query_buckets[i]) {
            result.candidate_matches += bucket.end - bucket.begin;
            for (std::size_t record_index = bucket.begin;
                 record_index < bucket.end; ++record_index) {
                const auto& indices = protein_triplet_index.records[record_index].indices;
                const std::array<interaction::point3D, 3> target_points{
                    pose_targets[indices[0]].position,
                    pose_targets[indices[1]].position,
                    pose_targets[indices[2]].position};
                for (std::size_t order_index = 0;
                     order_index < query.equivalent_order_count; ++order_index) {
                    const auto& order = query.equivalent_orders[order_index];
                    alignments.push_back({{
                        ligand_matching_areas[order[0]].centroid,
                        ligand_matching_areas[order[1]].centroid,
                        ligand_matching_areas[order[2]].centroid}, target_points});
                }
            }
        }
    }
    result.transforms_generated = alignments.size();
    append_stage("matching_expansion", start, alignments.size());

    // Allocation and initialization are excluded from the fitting batch.
    std::vector<RigidTransform> transforms(alignments.size());
    std::vector<unsigned char> valid(alignments.size());
    start = Clock::now();
    for (std::size_t i = 0; i < alignments.size(); ++i) {
        valid[i] = rigid_transform_from_points(
            alignments[i].source, alignments[i].target, transforms[i]);
    }
    append_stage("kabsch_fit", start, alignments.size());
    result.transforms_valid = static_cast<std::size_t>(
        std::count(valid.begin(), valid.end(), 1));

    const char* clash_method = std::getenv("DOCKING_CPU_CLASH_METHOD");
    const bool use_cached_clash = clash_method &&
        std::strcmp(clash_method, "cached") == 0;
    std::vector<interaction::point3D> ligand_points;
    start = Clock::now();
    if (use_cached_clash) {
        const auto& conformer = ligand.getConformer();
        ligand_points.reserve(ligand.getNumAtoms());
        for (unsigned int i = 0; i < ligand.getNumAtoms(); ++i) {
            const auto position = conformer.getAtomPos(i);
            ligand_points.push_back({
                static_cast<interaction::coordinate_type>(position.x),
                static_cast<interaction::coordinate_type>(position.y),
                static_cast<interaction::coordinate_type>(position.z)});
        }
    }
    append_stage("atom_preparation", start, ligand_points.size());

    std::vector<FastClashScore> clashes(alignments.size());
    start = Clock::now();
    for (std::size_t i = 0; i < transforms.size(); ++i) {
        if (!valid[i]) continue;
        clashes[i] = use_cached_clash
            ? score_fast_grid_clash_cached(ligand_points, transforms[i], protein_grid)
            : score_fast_grid_clash(ligand, transforms[i], protein_grid);
    }
    append_stage("atom_transform_clash", start, result.transforms_valid);

    std::vector<std::size_t> survivors;
    survivors.reserve(result.transforms_valid);
    start = Clock::now();
    for (std::size_t i = 0; i < transforms.size(); ++i) {
        if (!valid[i] || clashes[i].fraction > clash_thresholds.back()) continue;
        if (clashes[i].fraction > operational_clash_threshold) continue;
        survivors.push_back(i);
    }
    result.survivors = survivors.size();
    append_stage("survivor_selection", start, transforms.size());

    start = Clock::now();
    for (const auto index : survivors) {
        const auto transformed = transform_matching_areas(
            ligand_matching_areas, transforms[index]);
        const auto scored = score_matching_areas_weighted(transformed, protein_grid);
        if (!result.has_fast_score || scored.weighted_score > result.best_fast_score) {
            result.best_fast_score = scored.weighted_score;
            result.has_fast_score = true;
        }
    }
    append_stage("fast_scoring", start, survivors.size());
    // Parent interval includes setup and allocations above, but not buffer teardown.
    append_stage("replay_total", replay_start, alignments.size());
    return result;
}


PoseGenerationResult generate_candidate_poses(
    const RDKit::ROMol& ligand,
    const std::vector<MatchingArea>& ligand_matching_areas,
    const std::vector<Triplet>& ligand_triplets,
    const std::vector<PoseTarget>& pose_targets,
    const ProteinGrid& protein_grid,
    double ligand_diameter,
    double triplet_bin_size,
    double operational_clash_threshold,
    const std::array<double, 4>& clash_thresholds)
{
    using Clock =
        std::chrono::steady_clock;


    const auto index_start =
        Clock::now();


    const auto protein_triplet_hash =
        build_protein_triplet_hash_table(
            pose_targets,
            ligand_diameter,
            triplet_bin_size
        );
        
    const auto flat_triplet_index =
        flatten_protein_triplet_index(
            protein_triplet_hash
        );


    const auto index_end =
        Clock::now();


    auto result =
        generate_candidate_poses_from_index(
            ligand,
            ligand_matching_areas,
            ligand_triplets,
            pose_targets,
            protein_grid,
            flat_triplet_index,
            triplet_bin_size,
            operational_clash_threshold,
            clash_thresholds
        );


    result.timings
        .protein_triplet_index_seconds =
            std::chrono::duration<double>(
                index_end -
                index_start
            ).count();


    return result;
}
