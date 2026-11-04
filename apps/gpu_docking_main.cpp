// Configurable CUDA search with per-record ranking and pose export. The target
// persists across records; the rescoring backend does not change the search backend.
#include "gpu_docking_config.hpp"
#include "cuda_triplet_index.hpp"
#include "molecule_io.hpp"
#include "protein_area_crop.hpp"
#include "gpu_pose_search.hpp"
#include "gpu_ligand_prefetch.hpp"
#include "gpu_batch_retry.hpp"
#include "overlap_timing.hpp"
#include "geometry.hpp"

#include <cuda_runtime_api.h>
#include <GraphMol/SanitException.h>

#include <chrono>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t batch_prefetch_bytes = std::size_t{256} * 1024 * 1024;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string csv_string(const std::string& value) {
    std::string result = "\"";
    for (const char c : value) {
        if (c == '"') result += '"';
        result += c;
    }
    return result + '"';
}

void write_summary(std::ostream& output, const GpuDockingCliOptions& options,
                   const GpuDockingScreen& screen, std::size_t processed,
                   std::size_t successful_records, std::size_t failed_records,
                   double wall_before_summary, int query_threads_requested,
                   double online_wall_s, double consumer_wait_s, double producer_wait_s,
                   const OverlapStatistics& overlap, const GpuPrefetchStatistics& batch_queue,
                   std::size_t adaptive_splits, std::size_t memory_retries,
                   const OverlapStatistics& batch_overlap) {
    const auto& config = options.config;
    const auto& settings = options.settings;
    const auto method = settings.ranking.method;
    const bool weighted_interaction = method == PoseScoreMethod::WeightedInteraction || method == PoseScoreMethod::Hybrid;
    const bool vinardo = method == PoseScoreMethod::Vinardo || method == PoseScoreMethod::Hybrid;
    const auto crop_margin = protein_area_crop_margin();
    output << std::setprecision(17) << "key,value\n";
    const auto write = [&](const std::string& key, const auto& value) {
        std::ostringstream text;
        text << std::setprecision(17) << value;
        write_gpu_docking_summary_entry(output, key, text.str());
    };
    const auto environment = [](const char* key) {
        const char* value = std::getenv(key);
        return std::string(value ? value : "unset");
    };
    write("status", failed_records ? "complete_with_errors" : "complete");
    write("processed", processed);
    write("successful_records", successful_records);
    write("failed_records", failed_records);
    write("continue_on_error", options.continue_on_error);
    write("sdf_export", !options.no_sdf && options.pose_export.mode != PoseExportMode::None);
    write("top_k", options.top_k);
    write("pose_output", pose_export_mode_name(options.pose_export.mode));
    write("pose_detail", options.pose_export.detail == PoseExportDetail::Compact ? "compact" : "full");
    if (options.pose_output_controls) {
        write("pose_metrics", "pose_metrics.csv; Top-1 decision before export selection");
        if (!options.references.empty())
            write("best_rmsd_scope", "complete ranked post-prefilter pool; no pre-prefilter oracle available through this API");
    }
    write("limit", options.limit == std::numeric_limits<std::size_t>::max()
        ? "all" : std::to_string(options.limit));
    write("protein", config.protein_path.string());
    write("pocket", config.pocket_path.string());
    write("ligands", options.input.string());
    write("reference_mol2", options.references.string());
    write("ranking_requested", options.ranking_requested);
    write("ranking_method", pose_score_method_name(method));
    write("rescoring_backend", method == PoseScoreMethod::Fast ? "not_calculated" :
        settings.rescoring_backend == GpuRescoringBackend::Cuda ? "cuda" : "cpu");
    write("survivor_download_policy", options.gpu_batch && method != PoseScoreMethod::Fast
        ? "only geometry alignment IDs downloaded for segmentation; transforms/clash/Fast retained for CUDA rescoring; final ranking downloads requested rows"
        : settings.rescoring_backend == GpuRescoringBackend::Cuda && method != PoseScoreMethod::Fast ?
            (options.pose_export.mode == PoseExportMode::All ||
             (options.pose_output_controls && !options.references.empty())
                ? "all ranked poses after device rescoring; complete post-prefilter pool"
                : "top-k only after device rescoring and ranking") : "all survivors after Fast");
    write("ranking_direction", method == PoseScoreMethod::Vinardo ? "ascending" : "descending");
    if (method == PoseScoreMethod::Fast)
        write("ranking", "Fast descending; ties alignment_index ascending; no clustering");
    else
        write("ranking", std::string(pose_score_method_name(method)) +
            (method == PoseScoreMethod::Vinardo ? " ascending" : " descending") +
            "; ties original pose_id ascending; no clustering");
    write("tie_break", "original alignment/survivor order ascending");
    write("pose_id", "zero-based original clash-survivor ordinal; alignment_index retained separately");
    write("pose_pool", settings.fast_prefilter ? "explicit_fast_prefilter" : "all_clash_survivors");
    write("fast_prefilter", settings.fast_prefilter ? std::to_string(*settings.fast_prefilter) : "none");
    write("index_policy", "shared_index_only");
    write("index_cutoff_A", settings.cutoff);
    write("index_records", screen.index_records());
    write("grid_voxel_size_A", config.grid_voxel_size);
    write("grid_margin_A", config.grid_margin);
    write("triplet_bin_size_A", config.triplet_bin_size);
    write("pocket_area_max_distance_A", config.pocket_centroid_distance);
    write("operational_clash_fraction", config.operational_clash_threshold);
    write("target_clearance_enabled", settings.min_target_clearance.has_value());
    if (settings.min_target_clearance) write("min_target_clearance_A", *settings.min_target_clearance);
    else write("min_target_clearance_A", "off");
    write("targets_before_pruning", screen.targets_before_pruning());
    write("targets_after_pruning", screen.targets_after_pruning());
    write("kabsch_method_effective", options.kabsch_method);
    write("kabsch_source", options.kabsch_source);
    write("query_method_effective", options.query_method);
    write("query_source", options.query_source);
    write("query_threads_requested", query_threads_requested);
    write("query_openmp_available", gpu_query_openmp_available());
    write("query_parallel_min_triplets", gpu_query_parallel_min_triplets);
    write("query_min_triplets_per_requested_thread", gpu_query_min_per_thread);
    write("query_threads_used_source", "timings.csv per successful record when timings enabled; serial fallback uses one");
    write("DOCKING_KABSCH_METHOD", environment("DOCKING_KABSCH_METHOD"));
    write("DOCKING_QUERY_METHOD", environment("DOCKING_QUERY_METHOD"));
    write("DOCKING_QUERY_THREADS", environment("DOCKING_QUERY_THREADS"));
    write("diagnostics_effective", settings.diagnostic_stages ? "stages" : "off");
    write("diagnostics_source", options.diagnostics_source);
    write("DOCKING_PROFILE_STAGES", environment("DOCKING_PROFILE_STAGES"));
    write("DOCKING_STAGE_TIMINGS", environment("DOCKING_STAGE_TIMINGS"));
    write("DOCKING_KABSCH_TIMINGS", environment("DOCKING_KABSCH_TIMINGS"));
    write("timings_enabled", settings.collect_timings);
    if (options.pose_output_controls) {
        write("reference_evaluation_scope", "reference parsing and validation; full ranked-pool RMSD when a reference is supplied");
        write("pose_export_scope", "pose CSV/SDF and optional pose_metrics.csv; excludes records and summary writes");
    }
    write("timing_overlap_policy", options.gpu_batch
        ? (options.cpu_gpu_overlap.value_or(false)
            ? "one CPU producer with bounded multi-batch FIFO; one serialized aggregate CUDA consumer"
            : "serial CPU preparation; aggregated CUDA phases across each batch")
        : options.cpu_gpu_overlap.value_or(false)
        ? "one CPU producer and one serialized CUDA consumer; bounded next-record slot"
        : "sequential records; no pipelined CPU/GPU overlap across records");
    if (options.gpu_batch) {
        write("gpu_batch", "on");
        write("batch_size", options.batch_size);
        write("batch_memory_budget_bytes", options.batch_memory_bytes);
        write("batch_memory_scope", "conservative allocation estimate including explicit scoring buffers, ranking vectors and retained compaction scratch; excludes shared receptor, CUDA context, wrapper-local query/scan scratch and library-internal sort/reduce allocations; not a total VRAM cap");
        write("batch_timings", "batches.csv; shared search/scoring and complete docking call, not per-ligand kernel times");
        write("batch_adaptive_splits", adaptive_splits);
        write("batch_memory_retries", memory_retries);
        write("batch_attempts", "batch_attempts.csv; rejected budget attempts included in online/full-call wall, not successful stage totals");
        if (options.cpu_gpu_overlap.has_value()) {
            write("cpu_gpu_overlap", *options.cpu_gpu_overlap ? "on" : "off");
            write("batch_prefetch_capacity_records", batch_queue.capacity);
            write("batch_prefetch_budget_bytes", batch_queue.max_ready_bytes);
            write("batch_prefetch_high_water_records", batch_queue.high_water_records);
            write("batch_prefetch_high_water_bytes", batch_queue.high_water_bytes);
            write("batch_prefetch_memory_scope", "estimated queued payload; excludes RDKit internals, parsing temporaries and one in-flight preparation; not an RSS cap");
            write("consumer_wait_s", consumer_wait_s);
            write("producer_queue_wait_s", producer_wait_s);
            write("batch_query_dock_host_overlap_s", batch_overlap.query_dock_host_overlap_s);
            write("batch_query_dock_overlap_records", batch_overlap.query_dock_overlap_records);
            write("batch_overlap_scope", "CPU query versus successful completed batch dock host spans; inspect CUDA timeline for actual kernel overlap");
            write("nvtx_available", gpu_overlap_trace_available());
            write("overlap_trace", options.overlap_trace);
        }
        if (method != PoseScoreMethod::Fast)
            write("batch_rescoring_scope", "aggregated WeightedInteraction/Vinardo scoring, feature H2D and selection D2D; chemistry preparation, Fast prefilter, ranking and Hybrid normalization remain separate per ligand");
        write("online_wall_s", online_wall_s);
        write("online_successful_ligands_per_s", online_wall_s > 0 ? successful_records / online_wall_s : 0.0);
        write("online_scope", "input read through final record export; excludes receptor startup and summary writes");
    }
    if (options.cpu_gpu_overlap.has_value() && !options.gpu_batch) {
        write("cpu_gpu_overlap", *options.cpu_gpu_overlap ? "on" : "off");
        write("online_wall_s", online_wall_s);
        write("online_successful_ligands_per_s", online_wall_s > 0 ? successful_records / online_wall_s : 0.0);
        write("online_scope", "input read through final record export and producer join; excludes receptor startup, timeline and summary writes");
        write("consumer_wait_s", consumer_wait_s);
        write("producer_queue_wait_s", producer_wait_s);
        write("queue_capacity", 1);
        write("query_search_host_overlap_s", overlap.query_search_host_overlap_s);
        write("query_search_overlap_records", overlap.query_search_overlap_records);
        write("overlap_evidence_scope", method == PoseScoreMethod::Fast
            ? "CPU query vs another record completed search/Fast-selection host interval; not isolated kernels"
            : "CPU query vs another record completed geometry, CUDA rescoring/ranking or whole-dock host wrapper; not isolated kernels");
        if (method != PoseScoreMethod::Fast) {
            write("query_rescoring_host_overlap_s", overlap.query_rescoring_host_overlap_s);
            write("query_rescoring_overlap_records", overlap.query_rescoring_overlap_records);
            write("query_dock_host_overlap_s", overlap.query_dock_host_overlap_s);
            write("query_dock_overlap_records", overlap.query_dock_overlap_records);
            write("overlap_timeline_scope", "search is completed geometry wrapper; rescoring is completed CUDA rescoring/ranking wrapper; dock contains both and final cleanup");
            write("rescoring_completed_scope", "host packing, selection, H2D, CUDA scoring/normalization/ranking, result D2H and scorer-local cleanup; final device-pool cleanup belongs to dock");
            write("dock_completed_scope", "complete dock_prepared call; excludes producer preparation and record export; contains geometry and rescoring intervals");
        }
        write("nvtx_available", gpu_overlap_trace_available());
        write("overlap_trace", options.overlap_trace);
    }
    write("query_preparation_scope", options.gpu_batch
        ? "per-record host query construction; concatenation is included in the shared batch search"
        : options.cpu_gpu_overlap.has_value()
        ? "host query packing in producer; search_completed_s is query work plus completed CUDA wrapper work, not pipeline elapsed time"
        : "host query packing; included in search_completed_s");
    write("gpu_search_scope", options.gpu_batch
        ? "per-record search_completed_s and gpu_search_s are blank; batches.csv reports the shared completed search"
        : "search_completed_s minus query_preparation_s; completed mixed CUDA wrapper interval; not kernel time");
    write("timing_aggregation", options.gpu_batch
        ? "batch search and scoring are shared across records; rescoring contains its measured phases and is contained in the full batch docking call; do not add nested intervals"
        : "query_preparation_s plus gpu_search_s replaces search_completed_s; CPU hybrid_normalization_s is included in top_k_selection_s");
    write("timings_failed_records", "omitted from timings.csv; included in wall_before_summary_s");
    write("weighted_interaction_calculated", weighted_interaction);
    write("vinardo_calculated", vinardo);
    write("hybrid_calculated", method == PoseScoreMethod::Hybrid);
    write("interaction_region", weighted_interaction ? (settings.interaction_full_protein ? "protein" : "pocket") : "not_calculated");
    write("fast_region", crop_margin
        ? "cropped-protein chemical masks clipped to pocket grid and margin"
        : "whole-protein chemical masks clipped to pocket grid and margin");
    write("vinardo_region", vinardo ? "whole-protein supported heavy atoms; pair distance < 8 A" : "not_calculated");
    write("protein_crop", crop_margin
        ? "whole-residue PDB crop around pocket bounds for chemical areas only; full protein for occupancy, clash and Vinardo"
        : "none; full protein used for chemical areas, occupancy, clash and Vinardo");
    if (crop_margin) write("chemical_area_crop_margin_A", *crop_margin);
    else write("chemical_area_crop_margin_A", "off");
    write("DOCKING_AREA_CROP_MARGIN_A", environment("DOCKING_AREA_CROP_MARGIN_A"));
    write("clustering", "none; unavailable in shared API");
    if (vinardo) {
        write("vinardo_alpha", settings.ranking.vinardo_alpha);
        write("vinardo_adjustment", "V_alpha=V+(alpha-1)*repulsion");
        write("vinardo_skipped_receptor_atoms", screen.vinardo_skipped_receptor_atoms());
    } else write("vinardo_alpha", "not_calculated");
    if (method == PoseScoreMethod::Hybrid) {
        write("hybrid_lambda", settings.ranking.hybrid_lambda);
        write("hybrid_formula", "lambda*z(WeightedInteraction)+(1-lambda)*z(-V_alpha)");
        write("hybrid_normalization", "population std N; zero std gives zero z; no epsilon");
        write("hybrid_precision", settings.rescoring_backend == GpuRescoringBackend::Cuda
            ? "parallel scaled Welford in double; scalar reductions cross host; per-pose z and Hybrid on device"
            : "serial scaled Welford in host long double");
        write("hybrid_normalization_pool", "same ligand surviving explicit prefilter; before final top-k");
    } else write("hybrid_lambda", "not_calculated");
    write("nonfinite_scores", "error; no silent removal");
    write("vinardo_units", "experimental Vinardo-like; not validated affinity or kcal/mol");
    write("rmsd", options.references.empty() ? "disabled" :
          "direct heavy atoms; fixed atom order; no alignment or symmetry; never ranking");
    if (options.pose_center_distance) {
        write("pose_center_distance", "on");
        write("pose_center_reference_mode", options.center_reference.empty() ? "paired_reference_mol2" : "shared_reference_mol2");
        write("pose_center_reference", (options.center_reference.empty() ? options.references : options.center_reference).string());
        write("pose_center_distance_scope", "exported prefix only; heavy-atom centroid separation in angstroms; no alignment; not RMSD or ranking");
    }
    const FastScoreWeights fast{};
    write("fast_weight_hydrophobic", fast.hydrophobic);
    write("fast_weight_hydrogen_bond", fast.hydrogen_bond);
    write("fast_weight_halogen_bond", fast.halogen_bond);
    write("fast_weight_cation", fast.cation);
    write("fast_weight_anion", fast.anion);
    write("fast_weight_aromatic", fast.aromatic);
    write("fast_weight_metal", fast.metal);
    write("fast_weight_chelated", fast.chelated);
    if (weighted_interaction) {
        const WeightedInteractionWeights weights{};
        write("weighted_interaction_weight_hydrophobic", weights.hydrophobic);
        write("weighted_interaction_weight_pi_stacking", weights.pi_stacking);
        write("weighted_interaction_weight_hydrogen_bond", weights.hydrogen_bond);
        write("weighted_interaction_weight_halogen_bond", weights.halogen_bond);
        write("weighted_interaction_weight_cation_pi", weights.cation_pi);
        write("weighted_interaction_weight_ionic", weights.ionic);
        write("weighted_interaction_weight_metal", weights.metal);
    }
    if (settings.collect_timings)
        write("startup_s", screen.startup_seconds());
    write("wall_before_summary_s", wall_before_summary);
    write("successful_ligands_per_s", wall_before_summary > 0.0
        ? static_cast<double>(successful_records) / wall_before_summary : 0.0);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            write_gpu_docking_usage(std::cout); return 0;
        }
        const auto options = parse_gpu_docking_options(argc, argv);
        if (options.overlap_trace && !gpu_overlap_trace_available())
            throw std::runtime_error("NVTX is unavailable in this build; configure CUDA with the NVTX library before --overlap-trace");
        const int query_threads_requested = gpu_query_requested_threads();
        const auto& settings = options.settings;
        const bool timed = settings.collect_timings;
        const bool extended = settings.ranking.method != PoseScoreMethod::Fast;
        const bool export_poses = options.pose_export.mode != PoseExportMode::None;
        const bool pool_metrics = options.pose_output_controls && !options.references.empty();
        const auto ranking_count = options.pose_export.mode == PoseExportMode::All || pool_metrics
            ? std::numeric_limits<std::size_t>::max() : options.top_k;
        for (const auto& path : {options.config.protein_path, options.config.pocket_path, options.input})
            if (!fs::is_regular_file(path)) throw std::runtime_error("Input not found: " + path.string());
        std::unique_ptr<Mol2Reader> reader;
        if (!options.cpu_gpu_overlap.has_value() && !options.gpu_batch)
            reader = std::make_unique<Mol2Reader>(options.input);
        std::unique_ptr<Mol2Reader> reference_reader;
        if (!options.references.empty()) reference_reader = std::make_unique<Mol2Reader>(options.references);
        if (fs::exists(options.output)) throw std::runtime_error("Output already exists: " + options.output.string());
        std::optional<std::array<double, 3>> shared_center;
        if (options.pose_center_distance && !options.center_reference.empty()) {
            Mol2Reader center_reader(options.center_reference);
            std::string center_block, center_name;
            if (!center_reader.next(center_block, center_name))
                throw std::runtime_error("Center reference contains no MOL2 records: " + options.center_reference.string());
            const auto center_molecule = parse_ligand_mol2_block(center_block);
            shared_center = heavy_atom_centroid(*center_molecule);
            if (center_reader.next(center_block, center_name))
                throw std::runtime_error("Center reference must contain exactly one MOL2 record: " + options.center_reference.string());
        }
        apply_gpu_docking_environment(options);
        const auto run_start = Clock::now();
        GpuDockingScreen screen(options.config, settings);
        if (!fs::create_directories(options.output)) throw std::runtime_error("Cannot create output directory");
        std::ofstream poses, pose_metrics, records(options.output / "records.csv");
        if (export_poses) {
            poses.open(options.output / "poses.csv");
            poses.exceptions(std::ios::badbit | std::ios::failbit);
            write_gpu_docking_pose_header(poses, reference_reader != nullptr, extended, &options.pose_export,
                options.pose_center_distance);
        }
        if (options.pose_output_controls) {
            pose_metrics.open(options.output / "pose_metrics.csv");
            pose_metrics.exceptions(std::ios::badbit | std::ios::failbit);
            write_gpu_docking_pose_metrics_header(pose_metrics);
        }
        records.exceptions(std::ios::badbit | std::ios::failbit);
        records << "record_index,molecule_id,status,diameter_A,diameter_gt_cutoff,ligand_triplets,"
                   "candidate_matches,alignments,valid_transforms,survivors,exported_poses,sdf";
        if (extended) records << ",candidate_pool";
        if (options.continue_on_error) records << ",error";
        records << '\n';
        std::ofstream timings;
        if (timed) {
            timings.open(options.output / "timings.csv");
            timings.exceptions(std::ios::badbit | std::ios::failbit);
            timings << std::setprecision(17)
                << "record_index,read_parse_s,ligand_preparation_s,search_completed_s,"
                   "top_k_selection_s,reference_rmsd_export_s";
            if (extended) timings << ",fast_prefilter_s,weighted_interaction_scoring_s,vinardo_scoring_s,"
                "rescoring_chemical_preparation_s,rescoring_h2d_s,rescoring_score_preparation_s,"
                "hybrid_normalization_s,rescoring_top_k_d2h_s";
            timings << ",query_preparation_s,gpu_search_s,query_threads_requested,query_threads_used";
            if (options.pose_output_controls) timings << ",reference_evaluation_s,pose_export_s";
            if (extended && options.cpu_gpu_overlap.has_value() && !options.gpu_batch)
                timings << ",rescoring_completed_s,dock_completed_s";
            timings << '\n';
        }
        std::size_t processed = 0;
        std::size_t successful_records = 0, failed_records = 0;
        std::string block, name, reference_block, reference_name;
        std::vector<OverlapRecordTiming> overlap_records;
        std::vector<OverlapRecordTiming> batch_overlap_records;
        std::vector<std::size_t> batch_timeline_slots;
        double consumer_wait_s = 0.0, producer_wait_s = 0.0;
        std::size_t adaptive_splits = 0, memory_retries = 0, attempt_index = 0;
        // Start before the producer opens/reads the library. OFF and ON include
        // the same first-record warmup, final drain and per-record output.
        const auto online_start = options.cpu_gpu_overlap.has_value() || options.gpu_batch
            ? Clock::now() : Clock::time_point{};
        const auto relative_time = [&](Clock::time_point point) {
            return std::chrono::duration<double>(point - online_start).count();
        };
        std::unique_ptr<GpuLigandPrefetch> prefetch;
        if (options.cpu_gpu_overlap.has_value() && !options.gpu_batch)
            prefetch = std::make_unique<GpuLigandPrefetch>(options.input, options.config, options.limit,
                screen.index_records() != 0, *options.cpu_gpu_overlap, timed, options.overlap_trace);
        std::unique_ptr<GpuLigandPrefetch> batch_reader;
        std::vector<std::unique_ptr<GpuPreparedRecord>> batch_records;
        std::vector<GpuDockingResult> batch_results;
        std::size_t batch_cursor = 0, batch_index = 0;
        std::ofstream batches, attempts;
        if (options.gpu_batch) {
            batch_reader = std::make_unique<GpuLigandPrefetch>(options.input, options.config, options.limit,
                screen.index_records() != 0, options.cpu_gpu_overlap.value_or(false), timed, options.overlap_trace,
                options.cpu_gpu_overlap.has_value() ? 2 * options.batch_size : 1,
                options.cpu_gpu_overlap.has_value() ? batch_prefetch_bytes : 0);
            batches.open(options.output / "batches.csv");
            batches.exceptions(std::ios::badbit | std::ios::failbit);
            batches << std::setprecision(17)
                << "batch_index,records,prepared_records,candidate_matches,alignments,valid_transforms,"
                   "survivors,search_completed_s,dock_completed_s,explicit_device_bytes";
            if (extended)
                batches << ",candidate_pool,rescoring_completed_s,fast_prefilter_s,"
                    "weighted_interaction_scoring_s,vinardo_scoring_s,rescoring_chemical_preparation_s,"
                    "rescoring_h2d_s,rescoring_score_preparation_s,hybrid_normalization_s,rescoring_top_k_d2h_s";
            batches << '\n';
            attempts.open(options.output / "batch_attempts.csv");
            attempts.exceptions(std::ios::badbit | std::ios::failbit);
            attempts << std::setprecision(17)
                << "batch_index,attempt_index,first_record,last_record,prepared_records,status,"
                   "required_bytes,budget_bytes,dock_start_s,dock_end_s\n";
        }
        while (processed < options.limit) {
            const auto parse_start = timed ? Clock::now() : Clock::time_point{};
            std::unique_ptr<GpuPreparedRecord> ready;
            std::size_t index;
            if (batch_reader) {
                if (batch_cursor == batch_records.size()) {
                    batch_records.clear();
                    batch_results.clear();
                    batch_timeline_slots.clear();
                    batch_cursor = 0;
                    while (batch_records.size() < options.batch_size) {
                        double wait_s = 0.0;
                        auto packet = batch_reader->next(&wait_s);
                        consumer_wait_s += wait_s;
                        if (!packet) break;
                        if (options.cpu_gpu_overlap.has_value()) {
                            OverlapRecordTiming row;
                            row.record_index = packet->index;
                            row.status = packet->error ? "error" : "prepared";
                            if (packet->prepared.query_start != Clock::time_point{}) {
                                row.query_start_s = relative_time(packet->prepared.query_start);
                                row.query_end_s = relative_time(packet->prepared.query_end);
                            }
                            batch_timeline_slots.push_back(batch_overlap_records.size());
                            batch_overlap_records.push_back(std::move(row));
                        }
                        const bool stop_at_error = packet->error && !options.continue_on_error;
                        batch_records.push_back(std::move(packet));
                        if (stop_at_error) break;
                    }
                    if (batch_records.empty()) break;
                    std::vector<GpuPreparedLigand*> prepared;
                    std::vector<std::size_t> slots;
                    for (std::size_t i = 0; i < batch_records.size(); ++i) {
                        if (!batch_records[i]->error) {
                            prepared.push_back(&batch_records[i]->prepared);
                            slots.push_back(i);
                        }
                    }
                    GpuDockingBatchTimings batch_timings;
                    std::vector<std::exception_ptr> batch_errors(prepared.size());
                    const auto batch_start = timed ? Clock::now() : Clock::time_point{};
                    std::vector<GpuDockingResult> completed(prepared.size());
                    try {
                        run_gpu_batch_adaptive(0, prepared.size(), [&](auto first, auto last, bool single_retry) {
                            const auto started = Clock::now();
                            const auto attempt = ++attempt_index;
                            const auto log_attempt = [&](const char* status, std::size_t required, Clock::time_point ended) {
                                attempts << batch_index + 1 << ',' << attempt << ','
                                    << batch_records[slots[first]]->index << ','
                                    << batch_records[slots[last - 1]]->index << ',' << last - first
                                    << ',' << status << ',' << required << ',' << options.batch_memory_bytes
                                    << ',' << relative_time(started) << ',' << relative_time(ended) << '\n';
                                attempts.flush();
                            };
                            GpuDockingBatchTimings part;
                            std::vector<std::exception_ptr> errors;
                            std::vector<GpuDockingResult> results;
                            try {
                                std::vector<GpuPreparedLigand*> group(prepared.begin() + first, prepared.begin() + last);
                                GpuOverlapTraceRange trace(options.overlap_trace, "gpu_batch_dock");
                                results = screen.dock_batch_prepared(group, ranking_count,
                                    options.batch_memory_bytes, &part, &errors);
                            } catch (const GpuBatchMemoryError& error) {
                                ++memory_retries;
                                if (last - first > 1) ++adaptive_splits;
                                log_attempt(last - first > 1 ? "split" : single_retry ? "budget_error" : "retry_single",
                                    error.required_bytes, Clock::now());
                                if (single_retry)
                                    throw std::runtime_error("Single ligand record " +
                                        std::to_string(batch_records[slots[first]]->index) +
                                        " exceeds the explicit memory budget after releasing scratch: " + error.what());
                                throw;
                            }
                            const auto ended = Clock::now();
                            log_attempt("ok", part.explicit_device_bytes, ended);
                            if (results.size() != last - first || errors.size() != results.size())
                                throw std::runtime_error("GPU sub-batch output count mismatch");
                            batch_timings.search_seconds += part.search_seconds;
                            batch_timings.rescoring_seconds += part.rescoring_seconds;
                            batch_timings.explicit_device_bytes = std::max(batch_timings.explicit_device_bytes,
                                part.explicit_device_bytes);
                            for (auto field : {&GpuDockingTimings::prefilter_seconds,
                                    &GpuDockingTimings::weighted_interaction_seconds, &GpuDockingTimings::vinardo_seconds,
                                    &GpuDockingTimings::chemical_preparation_seconds, &GpuDockingTimings::rescoring_upload_seconds,
                                    &GpuDockingTimings::score_preparation_seconds, &GpuDockingTimings::hybrid_seconds,
                                    &GpuDockingTimings::selection_seconds, &GpuDockingTimings::rescoring_download_seconds})
                                batch_timings.rescoring.*field += part.rescoring.*field;
                            for (auto i = first; i < last; ++i) {
                                completed[i] = std::move(results[i - first]);
                                batch_errors[i] = errors[i - first];
                                if (options.cpu_gpu_overlap.has_value()) {
                                    auto& row = batch_overlap_records[batch_timeline_slots[slots[i]]];
                                    row.dock_start_s = relative_time(started);
                                    row.dock_end_s = relative_time(ended);
                                }
                            }
                        }, [&] { screen.release_batch_scratch(); });
                    } catch (const std::bad_alloc&) {
                        throw;
                    } catch (const std::exception& error) {
                        throw std::runtime_error("GPU batch starting at record " +
                            std::to_string(batch_records.front()->index) + ": " + error.what());
                    }
                    const double batch_seconds = timed ? elapsed(batch_start) : 0.0;
                    if (completed.size() != slots.size() || batch_errors.size() != slots.size())
                        throw std::runtime_error("GPU batch output count mismatch");
                    batch_results.resize(batch_records.size());
                    std::size_t matches = 0, alignments = 0, valid = 0, survivors = 0, candidate_pool = 0;
                    for (std::size_t i = 0; i < slots.size(); ++i) {
                        matches += completed[i].candidate_matches;
                        alignments += completed[i].alignments;
                        valid += completed[i].valid_transforms;
                        survivors += completed[i].survivors;
                        candidate_pool += completed[i].candidate_pool;
                        batch_results[slots[i]] = std::move(completed[i]);
                        if (batch_errors[i]) {
                            batch_records[slots[i]]->error = batch_errors[i];
                            batch_records[slots[i]]->error_stage = "preparation/search/ranking";
                        }
                    }
                    batches << ++batch_index << ',' << batch_records.size() << ',' << prepared.size()
                        << ',' << matches << ',' << alignments << ',' << valid << ',' << survivors << ',';
                    if (timed) batches << batch_timings.search_seconds;
                    batches << ',';
                    if (timed) batches << batch_seconds;
                    batches << ',' << batch_timings.explicit_device_bytes;
                    if (extended) {
                        batches << ',' << candidate_pool << ',';
                        if (timed) batches << batch_timings.rescoring_seconds;
                        for (const auto seconds : {batch_timings.rescoring.prefilter_seconds,
                                batch_timings.rescoring.weighted_interaction_seconds,
                                batch_timings.rescoring.vinardo_seconds,
                                batch_timings.rescoring.chemical_preparation_seconds,
                                batch_timings.rescoring.rescoring_upload_seconds,
                                batch_timings.rescoring.score_preparation_seconds,
                                batch_timings.rescoring.hybrid_seconds,
                                batch_timings.rescoring.rescoring_download_seconds}) {
                            batches << ',';
                            if (timed) batches << seconds;
                        }
                    }
                    batches << '\n';
                    batches.flush();
                }
                ready = std::move(batch_records[batch_cursor]);
                index = ready->index;
                name = ready->name;
            } else if (prefetch) {
                double wait_s = 0.0;
                ready = prefetch->next(&wait_s);
                consumer_wait_s += wait_s;
                if (!ready) break;
                index = ready->index;
                name = ready->name;
                OverlapRecordTiming row;
                row.record_index = index;
                row.consumer_wait_s = wait_s;
                if (ready->prepared.query_start != Clock::time_point{}) {
                    row.query_start_s = relative_time(ready->prepared.query_start);
                    row.query_end_s = relative_time(ready->prepared.query_end);
                }
                overlap_records.push_back(std::move(row));
            } else {
                if (!reader->next(block, name)) break;
                index = reader->index();
            }
            const char* record_stage = "parsing/sanitization";
            try {
                std::unique_ptr<RDKit::RWMol> parsed_molecule;
                RDKit::RWMol* molecule = nullptr;
                GpuDockingResult result;
                double parse_s = 0.0;
                bool exporting = false;
                try {
                    if (ready) {
                        record_stage = ready->error_stage;
                        if (ready->error) std::rethrow_exception(ready->error);
                        molecule = ready->prepared.molecule.get();
                        parse_s = ready->parse_seconds;
                    } else {
                        parsed_molecule = parse_ligand_mol2_block(block);
                        molecule = parsed_molecule.get();
                        parse_s = timed ? elapsed(parse_start) : 0.0;
                    }
                    record_stage = "preparation/search/ranking";
                    if (options.gpu_batch) {
                        result = std::move(batch_results[batch_cursor]);
                    } else if (ready) {
                        auto& span = overlap_records.back();
                        if (!extended) {
                            span.search_start_s = relative_time(Clock::now());
                            {
                                GpuOverlapTraceRange range(options.overlap_trace, "gpu_search_and_fast_selection");
                                result = screen.dock_prepared(ready->prepared, ranking_count);
                            }
                            span.search_end_s = relative_time(Clock::now());
                        } else {
                            span.dock_start_s = relative_time(Clock::now());
                            try {
                                GpuOverlapTraceRange range(options.overlap_trace, "gpu_docking_completed");
                                result = screen.dock_prepared(ready->prepared, ranking_count);
                            } catch (...) {
                                // Preserve the completed call span on recoverable record
                                // errors without inventing successful substage boundaries.
                                span.dock_end_s = relative_time(Clock::now());
                                throw;
                            }
                            span.dock_end_s = relative_time(Clock::now());
                            span.search_start_s = relative_time(result.timings.geometry_start);
                            span.search_end_s = relative_time(result.timings.geometry_end);
                            span.rescoring_start_s = relative_time(result.timings.rescoring_start);
                            span.rescoring_end_s = relative_time(result.timings.rescoring_end);
                        }
                    } else result = screen.dock(*molecule, ranking_count);
                    if (settings.diagnostic_stages)
                        write_stage_profile(std::cerr, "gpu_docking", index, result.stages);
                    const auto export_start = timed ? Clock::now() : Clock::time_point{};
                    std::unique_ptr<RDKit::RWMol> reference;
                    // Pair references by record position. Ranking has already finished;
                    // these coordinates are used only for atom mapping and RMSD export.
                    if (reference_reader) {
                        if (!reference_reader->next(reference_block, reference_name))
                            throw std::runtime_error("Missing paired reference MOL2 record");
                        reference = parse_ligand_mol2_block(reference_block);
                        validate_gpu_docking_reference(*molecule, *reference);
                    }
                    exporting = true;
                    record_stage = "reference RMSD/output";
                    std::vector<double> ranked_rmsds;
                    if (pool_metrics)
                        ranked_rmsds = gpu_docking_pose_rmsds(*molecule, result.poses, *reference);
                    std::vector<double> center_distances;
                    if (options.pose_center_distance && export_poses) {
                        if (!shared_center && !reference)
                            throw std::runtime_error("Pose center distance has no reference");
                        const auto center = shared_center ? *shared_center : heavy_atom_centroid(*reference);
                        center_distances = gpu_docking_pose_center_distances(*molecule, result.poses, center,
                            pose_export_count(options.pose_export, result.poses.size()));
                    }
                    const double evaluation_s = timed && options.pose_output_controls
                        ? elapsed(export_start) : 0.0;
                    const auto pose_export_start = timed && options.pose_output_controls
                        ? Clock::now() : Clock::time_point{};
                    const auto exported_count = pose_export_count(options.pose_export, result.poses.size());
                    const bool export_sdf = export_poses && !options.no_sdf;
                    const auto sdf_name = !export_sdf ? std::string{} :
                        "record-" + std::to_string(index) + ".sdf";
                    std::ofstream sdf;
                    if (export_sdf) {
                        sdf.open(options.output / sdf_name);
                        sdf.exceptions(std::ios::badbit | std::ios::failbit);
                    }
                    if (export_poses) {
                        record_stage = export_sdf ? "SDF export" : "CSV export";
                        write_gpu_docking_poses(poses, export_sdf ? &sdf : nullptr,
                            *molecule, result.poses, index, name, reference_reader != nullptr,
                            reference.get(), extended ? &result.scores : nullptr, &options.pose_export,
                            pool_metrics ? &ranked_rmsds : nullptr,
                            options.pose_center_distance ? &center_distances : nullptr);
                        if (export_sdf) sdf.close();
                        poses.flush();
                    }
                    if (options.pose_output_controls) {
                        write_gpu_docking_pose_metrics(pose_metrics, ranked_rmsds, index, name,
                            result.candidate_pool, result.poses.empty() ? nullptr : &result.poses.front(),
                            result.scores.empty() ? nullptr : &result.scores.front());
                        pose_metrics.flush();
                    }
                    const double export_s = timed ? elapsed(export_start) : 0.0;
                    const double pose_export_s = timed && options.pose_output_controls
                        ? elapsed(pose_export_start) : 0.0;
                    const char* status = result.ligand_triplets == 0 ? "no_triplets" :
                                         result.survivors == 0 ? "no_poses" : "ok";
                    if (prefetch) overlap_records.back().status = status;
                    records << std::setprecision(17) << index << ',' << csv_string(name) << ','
                        << status << ',' << result.diameter << ',' << (result.diameter > settings.cutoff) << ','
                        << result.ligand_triplets << ',' << result.candidate_matches << ','
                        << result.alignments << ',' << result.valid_transforms << ','
                        << result.survivors << ',' << exported_count << ',' << csv_string(sdf_name);
                    if (extended) records << ',' << result.candidate_pool;
                    if (options.continue_on_error) records << ',';
                    records << '\n'; records.flush();
                    if (timed) {
                        timings << index << ',' << parse_s << ',' << result.timings.preparation_seconds << ',';
                        if (!options.gpu_batch) timings << result.timings.search_seconds;
                        timings << ',' << result.timings.selection_seconds << ',' << export_s;
                        if (extended) {
                            timings << ',' << result.timings.prefilter_seconds << ',';
                            if (!options.gpu_batch) timings << result.timings.weighted_interaction_seconds;
                            timings << ',';
                            if (!options.gpu_batch) timings << result.timings.vinardo_seconds;
                            timings << ',' << result.timings.chemical_preparation_seconds << ',';
                            if (!options.gpu_batch) timings << result.timings.rescoring_upload_seconds;
                            timings << ',' << result.timings.score_preparation_seconds << ','
                                << result.timings.hybrid_seconds << ',' << result.timings.rescoring_download_seconds;
                        }
                        timings << ',' << result.timings.query_preparation_seconds << ',';
                        if (!options.gpu_batch) timings << result.timings.gpu_search_seconds;
                        timings << ',' << result.timings.query_threads_requested << ','
                            << result.timings.query_threads_used;
                        if (options.pose_output_controls) timings << ',' << evaluation_s << ',' << pose_export_s;
                        if (extended && prefetch) {
                            const auto& span = overlap_records.back();
                            timings << ',' << (*span.rescoring_end_s - *span.rescoring_start_s)
                                    << ',' << (*span.dock_end_s - *span.dock_start_s);
                        }
                        timings << '\n';
                        timings.flush();
                    }
                    ++successful_records;
                    if (options.gpu_batch && options.cpu_gpu_overlap.has_value())
                        batch_overlap_records[batch_timeline_slots[batch_cursor]].status = status;
                } catch (const GpuLigandSearchError&) {
                    throw; // A failed search must not become an expected dataset error.
                } catch (const GpuQueryPreparationError&) {
                    throw; // Moving query packing to a worker does not make its failures recoverable.
                } catch (const std::bad_alloc&) {
                    throw;
                } catch (const fs::filesystem_error&) {
                    throw;
                } catch (const std::ios_base::failure&) {
                    throw;
                } catch (const std::exception& error) {
                    if (!options.continue_on_error ||
                        (exporting && !dynamic_cast<const RDKit::MolSanitizeException*>(&error)))
                        throw;
                    const std::string reason = std::string(record_stage) + ": " + error.what();
                    // Only this consumer uses CUDA. Continue after cleanup has
                    // left a healthy, completed context for the next record.
                    const auto pending_error = cudaPeekAtLastError();
                    if (pending_error != cudaSuccess)
                        throw std::runtime_error(std::string("Cannot continue after CUDA failure: ") +
                            error.what() + "; " + cudaGetErrorString(pending_error));
                    const auto synchronized = cudaDeviceSynchronize();
                    if (synchronized != cudaSuccess) {
                        throw std::runtime_error(std::string("Cannot continue after record failure: ") +
                            error.what() + "; CUDA synchronization failed: " + cudaGetErrorString(synchronized));
                    }
                    records << index << ',' << csv_string(name)
                        << std::setprecision(17) << ",error," << result.diameter << ','
                        << (result.diameter > settings.cutoff) << ',' << result.ligand_triplets << ','
                        << result.candidate_matches << ',' << result.alignments << ','
                        << result.valid_transforms << ',' << result.survivors << ",0,\"\"";
                    if (extended) records << ',' << result.candidate_pool;
                    records << ',' << csv_string(reason) << '\n';
                    records.flush();
                    std::cerr << "Record " << index << " (" << name << "): " << reason << '\n';
                    ++failed_records;
                    if (prefetch) overlap_records.back().status = "error";
                    if (options.gpu_batch && options.cpu_gpu_overlap.has_value())
                        batch_overlap_records[batch_timeline_slots[batch_cursor]].status = "error";
                    ++processed;
                    if (options.gpu_batch) ++batch_cursor;
                    continue;
                }
            } catch (const std::exception& error) {
                // Fatal output/reference/resource errors and default fail-fast
                // failures retain partial output without a completed summary.
                throw std::runtime_error("Record " + std::to_string(index) + " (" + name +
                    ") [" + record_stage + "]: " + error.what());
            }
            ++processed;
            if (options.gpu_batch) ++batch_cursor;
        }
        if (prefetch) {
            prefetch->stop();
            producer_wait_s = prefetch->producer_queue_wait_seconds();
        }
        if (batch_reader) {
            batch_reader->stop();
            producer_wait_s = batch_reader->producer_queue_wait_seconds();
            batches.close();
            attempts.close();
        }
        const double online_wall_s = prefetch || batch_reader ? elapsed(online_start) : 0.0;
        if (processed == 0) throw std::runtime_error("Input contains no MOL2 records");
        if (reference_reader && processed < options.limit && reference_reader->next(reference_block, reference_name))
            throw std::runtime_error("Reference MOL2 has more records than the input");
        if (export_poses) poses.close();
        if (options.pose_output_controls) pose_metrics.close();
        records.close();
        if (timed) timings.close();
        OverlapStatistics overlap, batch_overlap;
        if (prefetch) {
            overlap = measure_host_overlap(overlap_records);
            std::ofstream timeline(options.output / "overlap_timeline.csv");
            timeline.exceptions(std::ios::badbit | std::ios::failbit);
            timeline << std::setprecision(17)
                << "record_index,status,query_start_s,query_end_s,search_start_s,search_end_s,consumer_wait_s";
            if (extended) timeline << ",dock_start_s,dock_end_s,rescoring_start_s,rescoring_end_s";
            timeline << '\n';
            for (const auto& row : overlap_records) {
                timeline << row.record_index << ',' << row.status;
                for (const auto& value : {row.query_start_s, row.query_end_s, row.search_start_s, row.search_end_s}) {
                    timeline << ',';
                    if (value) timeline << *value;
                }
                timeline << ',' << row.consumer_wait_s;
                if (extended)
                    for (const auto& value : {row.dock_start_s, row.dock_end_s, row.rescoring_start_s, row.rescoring_end_s}) {
                        timeline << ',';
                        if (value) timeline << *value;
                    }
                timeline << '\n';
            }
            timeline.close();
        }
        if (options.gpu_batch && options.cpu_gpu_overlap.has_value()) {
            std::ofstream timeline(options.output / "batch_overlap_timeline.csv");
            timeline.exceptions(std::ios::badbit | std::ios::failbit);
            timeline << std::setprecision(17)
                << "record_index,status,query_start_s,query_end_s,dock_start_s,dock_end_s\n";
            std::vector<OverlapRecordTiming> distinct_spans;
            std::optional<double> previous_dock;
            for (const auto& row : batch_overlap_records) {
                timeline << row.record_index << ',' << row.status;
                for (const auto& value : {row.query_start_s, row.query_end_s, row.dock_start_s, row.dock_end_s}) {
                    timeline << ',';
                    if (value) timeline << *value;
                }
                timeline << '\n';
                auto unique = row;
                if (unique.dock_start_s == previous_dock) {
                    unique.dock_start_s.reset();
                    unique.dock_end_s.reset();
                } else if (unique.dock_start_s) previous_dock = unique.dock_start_s;
                distinct_spans.push_back(std::move(unique));
            }
            timeline.close();
            batch_overlap = measure_host_overlap(distinct_spans);
        }
        const double wall_before_summary = elapsed(run_start);
        const auto summary_temporary = options.output / "summary.csv.tmp";
        std::ofstream summary(summary_temporary);
        summary.exceptions(std::ios::badbit | std::ios::failbit);
        write_summary(summary, options, screen, processed, successful_records, failed_records,
            wall_before_summary, query_threads_requested, online_wall_s,
            consumer_wait_s, producer_wait_s, overlap,
            batch_reader ? batch_reader->statistics() : GpuPrefetchStatistics{},
            adaptive_splits, memory_retries, batch_overlap);
        summary.close();
        fs::rename(summary_temporary, options.output / "summary.csv");
        std::cout << "Processed " << processed << " ligands";
        if (options.continue_on_error)
            std::cout << " (" << successful_records << " successful; " << failed_records << " failed)";
        std::cout << "; results: " << options.output << '\n';
        return failed_records ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'; return 1;
    }
}
