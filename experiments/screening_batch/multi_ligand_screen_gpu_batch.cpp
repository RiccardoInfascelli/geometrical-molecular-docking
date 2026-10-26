#include "cuda_screening_batch.hpp"
#include "cuda_pose_pipeline.hpp"
#include "cuda_geometry.hpp"
#include "cuda_triplet_index.hpp"
#include "ligand_preparation.hpp"
#include "molecule_io.hpp"
#include "protein_preparation.hpp"
#include "run_config.hpp"

#include <GraphMol/FileParsers/FileParsers.h>
#include <interaction/sanitize_mol.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void require_cuda(bool success, const char* operation) {
    if (!success) throw std::runtime_error(std::string("CUDA failed: ") + operation);
}

std::string csv_escape(const std::string& text) {
    std::string result = "\"";
    for (char c : text) {
        if (c == '"') result += '"';
        result += c;
    }
    return result + '"';
}

std::size_t positive_size(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected a positive integer: " + text);
    const auto value = std::stoull(text);
    if (!value || value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Integer out of range: " + text);
    return static_cast<std::size_t>(value);
}

struct Options {
    fs::path target, ligands, output;
    std::size_t limit = 30, batch_size = 16;
    double cutoff = 24.0;
    bool verify = false;
    GpuScreeningBatchOptions gpu;
};

Options parse_args(int argc, char** argv) {
    if (argc < 4) throw std::invalid_argument(
        "Usage: multi_ligand_screen_gpu_batch <target_complex_dir> <multi.mol2> "
        "<new_output_directory> [--limit 30] [--cutoff 24] [--batch-size 16] "
        "[--strategy warp|block|multi-block] [--block-size 128] [--blocks-per-ligand 8] "
        "[--memory-mib 256] [--verify]\n"
        "Requires DOCKING_KABSCH_METHOD=quaternion");
    Options opts;
    opts.target = argv[1]; opts.ligands = argv[2]; opts.output = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--verify") { opts.verify = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--limit") opts.limit = positive_size(value);
        else if (flag == "--batch-size") opts.batch_size = positive_size(value);
        else if (flag == "--block-size") {
            const auto size = positive_size(value);
            if (size > 1024 || size % 32 != 0)
                throw std::invalid_argument("--block-size must be a multiple of 32, at most 1024");
            opts.gpu.block_size = static_cast<std::uint32_t>(size);
        } else if (flag == "--blocks-per-ligand") {
            const auto blocks = positive_size(value);
            if (blocks > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("--blocks-per-ligand exceeds uint32 capacity");
            opts.gpu.blocks_per_ligand = static_cast<std::uint32_t>(blocks);
        } else if (flag == "--memory-mib") {
            const auto mib = positive_size(value);
            if (mib > std::numeric_limits<std::size_t>::max() / (1024 * 1024))
                throw std::invalid_argument("--memory-mib is too large");
            opts.gpu.max_device_bytes = mib * 1024 * 1024;
        } else if (flag == "--cutoff") {
            std::size_t end = 0;
            opts.cutoff = std::stod(value, &end);
            if (end != value.size() || !std::isfinite(opts.cutoff) || opts.cutoff <= 0)
                throw std::invalid_argument("--cutoff must be finite and positive");
        } else if (flag == "--strategy") {
            if (value == "warp") opts.gpu.strategy = GpuScreeningBatchStrategy::WarpPerLigand;
            else if (value == "block") opts.gpu.strategy = GpuScreeningBatchStrategy::BlockPerLigand;
            else if (value == "multi-block") opts.gpu.strategy = GpuScreeningBatchStrategy::MultiBlockPerLigand;
            else throw std::invalid_argument("--strategy must be warp, block or multi-block");
        } else throw std::invalid_argument("Unknown option: " + flag);
    }
    if (opts.batch_size > 1024) throw std::invalid_argument("--batch-size must be at most 1024");
    const char* method = std::getenv("DOCKING_KABSCH_METHOD");
    if (!method || std::strcmp(method, "quaternion") != 0)
        throw std::invalid_argument("This experiment requires DOCKING_KABSCH_METHOD=quaternion");
    if (!fs::is_regular_file(opts.ligands)) throw std::runtime_error("Input MOL2 not found");
    if (fs::exists(opts.output)) throw std::runtime_error("Output directory already exists");
    return opts;
}

// One target owner for the whole run, including optional baseline verification.
struct FixedTarget {
    GpuFlatTripletIndexDevice index{};
    GpuPointArrayDevice points{};
    GpuProteinGridDevice grid{};
    double preparation_s = 0, index_s = 0, upload_s = 0;
    std::size_t owned_bytes = 0;

    FixedTarget(const RunConfig& config, double cutoff) {
        try {
            auto start = Clock::now();
            const auto protein = prepare_protein(config);
            preparation_s = seconds_since(start);
            start = Clock::now();
            const auto hash = build_protein_triplet_hash_table(
                protein.targets, cutoff, config.triplet_bin_size);
            const auto packed_index = make_gpu_flat_triplet_index(flatten_protein_triplet_index(hash));
            index_s = seconds_since(start);
            start = Clock::now();
            require_cuda(cuda_upload_triplet_index(packed_index, index), "shared index upload");
            require_cuda(cuda_upload_points(make_gpu_pose_target_points(protein.targets), points),
                         "shared targets upload");
            require_cuda(cuda_upload_protein_grid(make_gpu_protein_grid(protein.grid), grid),
                         "shared grid upload");
            require_cuda(cudaDeviceSynchronize() == cudaSuccess, "target upload completion");
            upload_s = seconds_since(start);
            owned_bytes = std::size_t(index.key_count) * sizeof(GpuTripletKey)
                        + std::size_t(index.offset_count) * sizeof(std::uint32_t)
                        + std::size_t(index.record_count) * sizeof(GpuProteinTripletRecord)
                        + std::size_t(points.count) * sizeof(GpuPoint3)
                        + std::size_t(grid.voxel_count) * sizeof(GpuGridVoxel);
        } catch (...) { release(); throw; }
    }
    FixedTarget(const FixedTarget&) = delete;
    FixedTarget& operator=(const FixedTarget&) = delete;
    ~FixedTarget() { release(); }
    void release() noexcept {
        cuda_free_protein_grid(grid); cuda_free_points(points); cuda_free_triplet_index(index);
    }
};

struct Record {
    std::size_t index = 0, atoms = 0, areas = 0, matching_areas = 0, triplets = 0;
    std::string name, status = "ok", error;
    double diameter = 0, preparation_s = 0, query_pack_s = 0;
};

// Verification alone materializes the baseline's candidate arrays and survivor
// list. It runs after the measured batch, using the same packed ligand inputs.
struct ReferenceBuffers {
    GpuTripletNeighborCountArrayDevice counts{};
    GpuTripletNeighborOffsetArrayDevice offsets{};
    GpuTripletCandidateArrayDevice candidates{};
    GpuAlignmentCountArrayDevice alignment_counts{};
    GpuAlignmentOffsetArrayDevice alignment_offsets{};
    GpuAlignmentCandidateArrayDevice alignments{};
    GpuPointArrayDevice points{}, atoms{};
    GpuFastMatchingAreaArrayDevice areas{};
    ~ReferenceBuffers() {
        cuda_free_fast_matching_areas(areas); cuda_free_points(atoms); cuda_free_points(points);
        cuda_free_alignment_candidates(alignments); cuda_free_alignment_offsets(alignment_offsets);
        cuda_free_alignment_counts(alignment_counts); cuda_free_triplet_candidates(candidates);
        cuda_free_triplet_neighbor_offsets(offsets); cuda_free_triplet_neighbor_counts(counts);
    }
};

bool close(double a, double b, double tolerance) {
    return a == b || (std::isnan(a) && std::isnan(b)) || std::abs(a - b) <= tolerance;
}

void verify_result(const FixedTarget& target, const GpuScreeningBatchLigand& ligand,
                   const GpuScreeningBatchResult& batch, double threshold) {
    ReferenceBuffers device;
    GpuPosePipelineResult reference;
    if (!ligand.queries.empty()) {
        require_cuda(cuda_count_triplet_neighbor_records_device(target.index, ligand.queries,
                     device.counts), "reference lookup count");
        require_cuda(cuda_exclusive_triplet_neighbor_prefix_sum_device(device.counts,
                     device.offsets), "reference lookup scan");
        require_cuda(cuda_expand_triplet_neighbor_records_device(target.index, ligand.queries,
                     device.counts, device.offsets, device.candidates), "reference matching");
        require_cuda(cuda_count_alignment_candidates_device(device.candidates, ligand.orders,
                     device.alignment_counts), "reference alignment count");
        require_cuda(cuda_exclusive_prefix_sum_device(device.alignment_counts,
                     device.alignment_offsets), "reference alignment scan");
        require_cuda(cuda_expand_alignment_candidates_device(device.candidates, ligand.orders,
                     device.alignment_offsets, device.alignments), "reference alignment expansion");
        require_cuda(cuda_upload_points(ligand.matching_points, device.points), "reference points");
        require_cuda(cuda_upload_points(ligand.atoms, device.atoms), "reference atoms");
        require_cuda(cuda_upload_fast_matching_areas(ligand.fast_areas, device.areas), "reference areas");
        require_cuda(cuda_run_pose_pipeline(device.points, target.points, device.atoms, device.areas,
                     target.grid, device.alignments, threshold, reference), "reference pose pipeline");
    }
    const GpuPipelinePose* top = nullptr;
    for (const auto& pose : reference.poses)
        if (pose.fast_score.valid &&
            (!top || pose.fast_score.weighted_score > top->fast_score.weighted_score)) top = &pose;
    if (batch.candidate_matches != device.candidates.count ||
        batch.transforms_generated != device.alignments.count ||
        batch.valid_transforms != reference.valid_transform_count ||
        batch.survivors != reference.clash_survivor_count ||
        batch.has_top_pose != (top != nullptr))
        throw std::runtime_error("Batch/baseline count mismatch: matches " +
            std::to_string(batch.candidate_matches) + "/" + std::to_string(device.candidates.count) +
            ", transforms " + std::to_string(batch.transforms_generated) + "/" +
            std::to_string(device.alignments.count) + ", valid " + std::to_string(batch.valid_transforms) +
            "/" + std::to_string(reference.valid_transform_count) + ", survivors " +
            std::to_string(batch.survivors) + "/" + std::to_string(reference.clash_survivor_count));
    if (!top) return;
    if (!close(batch.top_fast_score.weighted_score, top->fast_score.weighted_score, 1e-9) ||
        batch.top_clash.occupied_atoms != top->clash.occupied_atoms)
        throw std::runtime_error("Batch/baseline top-1 score or clash mismatch");
    for (int i = 0; i < 9; ++i)
        if (!close(batch.top_transform.rotation[i], top->transform.rotation[i], 1e-8))
            throw std::runtime_error("Batch/baseline top-1 rotation mismatch");
    for (int i = 0; i < 3; ++i)
        if (!close(batch.top_transform.translation[i], top->transform.translation[i], 1e-8))
            throw std::runtime_error("Batch/baseline top-1 translation mismatch");
}

void write_record(std::ostream& csv, const Record& record,
                  const GpuScreeningBatchResult& result, std::uint32_t index_records) {
    csv << record.index << ',' << csv_escape(record.name) << ',' << record.status << ','
        << csv_escape(record.error) << ',' << record.atoms << ',' << record.areas << ','
        << record.matching_areas << ',' << record.diameter << ',' << record.triplets
        << ",shared," << index_records << ',' << result.candidate_matches << ','
        << result.transforms_generated << ',' << result.valid_transforms << ',' << result.survivors << ',';
    if (result.has_top_pose) csv << result.top_fast_score.weighted_score;
    csv << ',' << record.preparation_s << ',' << record.query_pack_s;
    if (result.has_top_pose) {
        csv << ',' << result.top_query_index << ',' << result.top_neighbor_slot << ','
            << result.top_record_index << ',' << result.top_order_index;
        for (double x : result.top_transform.rotation) csv << ',' << x;
        for (double x : result.top_transform.translation) csv << ',' << x;
    } else {
        for (int i = 0; i < 16; ++i) csv << ',';
    }
    csv << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const auto process_start = Clock::now();
    try {
        auto opts = parse_args(argc, argv);
        char program[] = "multi_ligand_screen_gpu_batch";
        auto target_path = opts.target.string();
        char* target_argv[] = {program, target_path.data()};
        const auto config = parse_run_config(2, target_argv);
        opts.gpu.clash_threshold = config.operational_clash_threshold;
        int device_id = 0, runtime_version = 0, driver_version = 0;
        cudaDeviceProp device_properties{};
        require_cuda(cudaGetDevice(&device_id) == cudaSuccess, "current device");
        require_cuda(cudaGetDeviceProperties(&device_properties, device_id) == cudaSuccess,
                     "device properties");
        require_cuda(cudaRuntimeGetVersion(&runtime_version) == cudaSuccess, "runtime version");
        require_cuda(cudaDriverGetVersion(&driver_version) == cudaSuccess, "driver version");
        if (!fs::create_directories(opts.output)) throw std::runtime_error("Cannot create new output directory");
        std::ofstream csv(opts.output / "records.csv"), batches(opts.output / "batches.csv");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        batches.exceptions(std::ios::badbit | std::ios::failbit);
        csv << std::setprecision(17)
            << "record_index,molecule_id,status,error,atoms,interaction_areas,matching_areas,"
               "diameter_A,ligand_triplets,protein_index_mode,protein_triplet_records,candidate_matches,"
               "transforms_generated,valid_transforms,survivors,best_fast_score,ligand_prep_s,query_pack_s,"
               "top_query_index,top_neighbor_slot,top_record_index,top_order_index,"
               "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n";
        batches << std::setprecision(17)
            << "batch_index,records,ligand_prep_s,query_pack_s,host_pack_s,allocation_s,h2d_s,"
               "gpu_kernel_s,d2h_s,host_reduction_s,cleanup_s,batch_call_s,batch_call_residual_s,"
               "online_wall_s,batch_device_bytes,h2d_bytes,d2h_bytes,sampled_device_used_bytes\n";
        const FixedTarget target(config, opts.cutoff);
        Mol2Reader reader(opts.ligands);
        std::size_t processed = 0, errors = 0, no_triplets = 0, oversized = 0, batch_index = 0;
        std::size_t peak_owned = target.owned_bytes, sampled_peak = 0;
        double prep_s = 0, query_s = 0, batch_s = 0, kernel_s = 0, h2d_s = 0, d2h_s = 0;
        double online_s = 0, verification_s = 0;
        bool finished = false;
        while (!finished && processed < opts.limit) {
            const auto online_start = Clock::now();
            std::vector<Record> records;
            std::vector<GpuScreeningBatchLigand> inputs;
            records.reserve(opts.batch_size); inputs.reserve(opts.batch_size);
            double batch_prep_s = 0, batch_query_s = 0;
            for (std::size_t n = 0; n < opts.batch_size && processed + records.size() < opts.limit; ++n) {
                const auto start = Clock::now();
                std::string block, name;
                if (!reader.next(block, name)) { finished = true; break; }
                Record record;
                record.index = reader.index(); record.name = name;
                GpuScreeningBatchLigand input;
                PreparedLigand prepared;
                try {
                    std::unique_ptr<RDKit::RWMol> molecule(RDKit::Mol2BlockToMol(block, false, false));
                    if (!molecule) throw std::runtime_error("RDKit returned null");
                    interaction::sanitize_molecule(molecule);
                    record.atoms = molecule->getNumAtoms();
                    prepare_ligand(*molecule, prepared);
                    if (prepared.diameter > opts.cutoff) ++oversized;
                    record.preparation_s = seconds_since(start);
                    const auto query_start = Clock::now();
                    make_gpu_ligand_triplet_queries(prepared.triplets, config.triplet_bin_size,
                                                   input.queries, input.orders);
                    input.matching_points = make_gpu_ligand_matching_points(prepared.matching_areas);
                    input.atoms = make_gpu_ligand_atom_points(*molecule);
                    input.fast_areas = make_gpu_fast_matching_areas(prepared.matching_areas, FastScoreWeights{});
                    record.query_pack_s = seconds_since(query_start);
                    if (prepared.triplets.empty()) { record.status = "no_triplets"; ++no_triplets; }
                } catch (const std::exception& e) {
                    record.status = "error"; record.error = e.what(); ++errors;
                    if (record.preparation_s == 0) record.preparation_s = seconds_since(start);
                    input = GpuScreeningBatchLigand{};
                }
                record.areas = prepared.areas.size(); record.matching_areas = prepared.matching_areas.size();
                record.diameter = prepared.diameter; record.triplets = prepared.triplets.size();
                batch_prep_s += record.preparation_s; batch_query_s += record.query_pack_s;
                records.push_back(std::move(record)); inputs.push_back(std::move(input));
            }
            if (records.empty()) break;
            GpuScreeningBatchTimings timings;
            std::vector<GpuScreeningBatchResult> results;
            std::string error;
            const auto call_start = Clock::now();
            if (!cuda_screen_ligand_batch(target.index, target.points, target.grid, inputs,
                                         opts.gpu, results, timings, error))
                throw std::runtime_error("Batch " + std::to_string(batch_index + 1) + ": " + error);
            const double call_s = seconds_since(call_start);
            const double batch_online_s = seconds_since(online_start);
            const auto verification_start = Clock::now();
            if (opts.verify) {
                for (std::size_t i = 0; i < records.size(); ++i) {
                    try { verify_result(target, inputs[i], results[i], opts.gpu.clash_threshold); }
                    catch (const std::exception& e) {
                        throw std::runtime_error("Verification record " + std::to_string(records[i].index) +
                                                 ": " + e.what());
                    }
                }
                verification_s += seconds_since(verification_start);
            }
            for (std::size_t i = 0; i < records.size(); ++i)
                write_record(csv, records[i], results[i], target.index.record_count);
            csv.flush();
            const std::size_t used = timings.total_device_bytes - timings.min_free_device_bytes;
            sampled_peak = std::max(sampled_peak, used);
            peak_owned = std::max(peak_owned, target.owned_bytes + timings.batch_device_bytes);
            const double child_s = timings.pack_seconds + timings.allocation_seconds + timings.h2d_seconds +
                timings.kernel_seconds + timings.d2h_seconds + timings.reduction_seconds + timings.cleanup_seconds;
            batches << ++batch_index << ',' << records.size() << ',' << batch_prep_s << ',' << batch_query_s
                << ',' << timings.pack_seconds << ',' << timings.allocation_seconds << ',' << timings.h2d_seconds
                << ',' << timings.kernel_seconds << ',' << timings.d2h_seconds << ',' << timings.reduction_seconds
                << ',' << timings.cleanup_seconds << ',' << call_s << ',' << call_s - child_s << ','
                << batch_online_s << ',' << timings.batch_device_bytes << ',' << timings.h2d_bytes << ','
                << timings.d2h_bytes << ',' << used << '\n';
            batches.flush();
            prep_s += batch_prep_s; query_s += batch_query_s; batch_s += call_s;
            kernel_s += timings.kernel_seconds; h2d_s += timings.h2d_seconds; d2h_s += timings.d2h_seconds;
            online_s += batch_online_s; processed += records.size();
            std::cerr << "Processed " << processed << '/' << opts.limit << ", errors=" << errors << '\n';
        }
        csv.close(); batches.close();
        const double wall_s = seconds_since(process_start);
        std::ofstream summary(opts.output / "summary.csv");
        summary.exceptions(std::ios::badbit | std::ios::failbit);
        const char* query_method = std::getenv("DOCKING_QUERY_METHOD");
        const bool warp = opts.gpu.strategy == GpuScreeningBatchStrategy::WarpPerLigand;
        const bool multi_block = opts.gpu.strategy == GpuScreeningBatchStrategy::MultiBlockPerLigand;
        const auto threads_per_block = warp ? 32u : opts.gpu.block_size;
        const auto blocks_per_ligand = multi_block ? opts.gpu.blocks_per_ligand : 1u;
        summary << std::setprecision(17) << "key,value\n"
            << "strategy," << (warp ? "warp" : multi_block ? "multi-block" : "block")
            << "\nthreads_per_ligand," << std::size_t(blocks_per_ligand) * threads_per_block
            << "\nthreads_per_block," << threads_per_block
            << "\nblocks_per_ligand," << blocks_per_ligand
            << "\nbatch_size," << opts.batch_size << "\nbatch_memory_budget_bytes," << opts.gpu.max_device_bytes
            << "\ncomplex," << csv_escape(config.complex_id) << "\ninput_mol2," << csv_escape(opts.ligands.string())
            << "\nlimit," << opts.limit << "\ncutoff_A," << opts.cutoff << "\nindex_policy,shared_index_only"
            << "\ngpu_device," << device_id << "\ngpu_name," << csv_escape(device_properties.name)
            << "\ncompute_capability," << device_properties.major << '.' << device_properties.minor
            << "\ncuda_runtime_version," << runtime_version << "\ncuda_driver_version," << driver_version
            << "\nprotein_triplet_records," << target.index.record_count
            << "\nprotein_index_keys," << target.index.key_count
            << "\ntriplet_bin_size_A," << config.triplet_bin_size << "\nclash_threshold," << opts.gpu.clash_threshold
            << "\nkabsch_method,quaternion\nquery_method," << csv_escape(query_method ? query_method : "reference")
            << "\nverification_enabled," << opts.verify << "\nprocessed," << processed
            << "\nok_or_no_triplets," << processed - errors << "\nno_triplets," << no_triplets
            << "\nerrors," << errors << "\nligands_diameter_gt_cutoff," << oversized
            << "\ntarget_preparation_s," << target.preparation_s << "\nshared_index_build_s," << target.index_s
            << "\nshared_gpu_upload_s," << target.upload_s << "\nligand_parse_preparation_s," << prep_s
            << "\nquery_pack_s," << query_s << "\nbatch_calls_s," << batch_s << "\ngpu_kernel_s," << kernel_s
            << "\nh2d_s," << h2d_s << "\nd2h_s," << d2h_s << "\nonline_wall_s," << online_s
            << "\nverification_s," << verification_s << "\nwall_total_s," << wall_s
            << "\nligands_per_s_online," << (online_s > 0 ? (processed - errors) / online_s : 0)
            << "\nligands_per_s_end_to_end," << (wall_s > 0 ? (processed - errors) / wall_s : 0)
            << "\nshared_owned_device_bytes," << target.owned_bytes
            << "\npeak_owned_device_bytes," << peak_owned << "\npeak_sampled_device_used_bytes," << sampled_peak << '\n';
        summary.close();
        std::cout << "processed=" << processed << "\nerrors=" << errors << "\nonline_wall_s=" << online_s
                  << "\nwall_total_s=" << wall_s << "\nresults_directory=" << opts.output << '\n';
        return errors ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
