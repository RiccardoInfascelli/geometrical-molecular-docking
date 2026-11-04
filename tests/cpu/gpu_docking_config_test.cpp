#include "gpu_docking_config.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()>& operation) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument");
}

std::optional<std::string> environment(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::optional<std::string>(value) : std::nullopt;
}

void environment(const char* name, const std::optional<std::string>& value) {
    const int status = value ? ::setenv(name, value->c_str(), 1) : ::unsetenv(name);
    if (status != 0) throw std::runtime_error(std::string("Could not update test environment: ") + name);
}

struct EnvironmentGuard {
    std::vector<std::pair<std::string, std::optional<std::string>>> original;
    EnvironmentGuard() {
        for (const char* name : {"DOCKING_KABSCH_METHOD", "DOCKING_QUERY_METHOD",
                                 "DOCKING_PROFILE_STAGES", "DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS"}) {
            original.emplace_back(name, environment(name));
            environment(name, std::nullopt);
        }
    }
    ~EnvironmentGuard() {
        for (const auto& [name, value] : original) {
            if (value) ::setenv(name.c_str(), value->c_str(), 1);
            else ::unsetenv(name.c_str());
        }
    }
};

GpuDockingCliOptions parse(const std::vector<std::string>& flags = {}) {
    std::vector<std::string> arguments = {"gpu_docking", "missing/protein.pdb", "missing/pocket.pdb",
                                          "missing/ligands.mol2", "uncreated/output"};
    arguments.insert(arguments.end(), flags.begin(), flags.end());
    std::vector<char*> argv;
    for (auto& value : arguments) argv.push_back(value.data());
    return parse_gpu_docking_options(static_cast<int>(argv.size()), argv.data());
}

void test_defaults_and_values() {
    const auto defaults = parse();
    require(defaults.config.protein_path == "missing/protein.pdb" &&
            defaults.config.pocket_path == "missing/pocket.pdb" &&
            defaults.input == "missing/ligands.mol2" && defaults.output == "uncreated/output",
            "Positional paths differ");
    require(defaults.config.ligand_path == defaults.input &&
            defaults.config.output_directory == defaults.output &&
            defaults.config.output_mode == OutputMode::Quiet, "RunConfig paths/mode differ");
    require(defaults.top_k == 20 && defaults.limit == std::numeric_limits<std::size_t>::max(),
            "Default record/pose limits differ");
    require(!defaults.continue_on_error, "Per-record error recovery became enabled by default");
    require(!defaults.no_sdf, "SDF export became disabled by default");
    require(!defaults.pose_output_controls && defaults.pose_export.mode == PoseExportMode::TopK &&
            defaults.pose_export.top_k == 20 && !defaults.pose_export.detail,
            "Default pose export policy changed");
    require(defaults.settings.cutoff == 24.0 && defaults.ranking_requested == "fast" &&
            defaults.settings.ranking.method == PoseScoreMethod::Fast &&
            defaults.settings.rescoring_backend == GpuRescoringBackend::Cpu &&
            !defaults.settings.collect_timings && !defaults.settings.diagnostic_stages &&
            !defaults.settings.min_target_clearance && !defaults.settings.fast_prefilter,
            "Default scientific settings differ");
    require(defaults.kabsch_method == "quaternion" && defaults.query_method == "optimized" &&
            defaults.kabsch_source == "default" && defaults.query_source == "default" &&
            defaults.diagnostics_source == "default", "Default selectors/sources differ");
    const auto reference = parse({"--kabsch", "svd", "--query-method", "reference"});
    require(reference.kabsch_method == "svd" && reference.query_method == "reference" &&
            reference.kabsch_source == "cli" && reference.query_source == "cli",
            "Explicit SVD/reference selection was not retained");
    const auto chosen = parse({"--top-k", "5", "--limit", "7", "--cutoff", "18.5", "--timings",
        "--reference-mol2", "missing/references.mol2", "--ranking", "hybrid", "--kabsch", "quaternion",
        "--query-method", "optimized", "--clash-threshold", "1", "--min-target-clearance", "0",
        "--interaction-region", "protein", "--fast-prefilter", "32", "--vinardo-alpha", ".5",
        "--hybrid-lambda", ".6", "--diagnostics", "stages"});
    require(chosen.top_k == 5 && chosen.limit == 7 && chosen.settings.cutoff == 18.5 &&
            chosen.settings.collect_timings && chosen.references == "missing/references.mol2",
            "Requested basic values differ");
    require(chosen.settings.ranking.method == PoseScoreMethod::Hybrid &&
            chosen.settings.ranking.vinardo_alpha == .5 && chosen.settings.ranking.hybrid_lambda == .6 &&
            chosen.settings.interaction_full_protein && chosen.settings.fast_prefilter == 32 &&
            chosen.config.operational_clash_threshold == 1.0, "Requested scientific values differ");
    require(chosen.settings.min_target_clearance.has_value() && *chosen.settings.min_target_clearance == 0.0,
            "Zero clearance became disabled");
    require(chosen.settings.diagnostic_stages && chosen.kabsch_method == "quaternion" &&
            chosen.query_method == "optimized" && chosen.kabsch_source == "cli" &&
            chosen.query_source == "cli" && chosen.diagnostics_source == "cli", "CLI selectors/sources differ");
    require(!environment("DOCKING_KABSCH_METHOD") && !environment("DOCKING_QUERY_METHOD") &&
            !environment("DOCKING_PROFILE_STAGES"), "Parser mutated the environment");
    require(!parse({"--min-target-clearance", "off"}).settings.min_target_clearance,
            "Explicit OFF clearance became active");
    require(parse({"--clash-threshold", "0"}).config.operational_clash_threshold == 0.0,
            "Clash threshold lower endpoint rejected");
    const auto continued = parse({"--continue-on-error", "--limit", "7"});
    require(continued.continue_on_error && continued.limit == 7,
            "Continue-on-error flag was not enabled or consumed the next option");
    const auto csv_only = parse({"--no-sdf", "--continue-on-error", "--timings", "--kabsch", "quaternion"});
    require(csv_only.no_sdf && csv_only.continue_on_error && csv_only.settings.collect_timings &&
            csv_only.kabsch_method == "quaternion", "CSV-only options changed another setting");
}

void test_pose_output_options() {
    for (const char* mode : {"none", "topk", "all"}) {
        const auto options = parse({"--pose-output", mode, "--top-k", "7", "--pose-detail", "compact",
                                   "--ranking", "hybrid", "--vinardo-alpha", ".5", "--hybrid-lambda", ".3"});
        require(options.pose_output_controls && options.pose_export.mode == parse_pose_export_mode(mode) &&
                options.top_k == 7 && options.pose_export.top_k == 7 &&
                options.pose_export.detail == PoseExportDetail::Compact,
                "Pose output options were not retained");
        require(!options.settings.fast_prefilter && options.settings.ranking.vinardo_alpha == .5 &&
                options.settings.ranking.hybrid_lambda == .3 &&
                options.settings.ranking.method == PoseScoreMethod::Hybrid,
                "Pose output options changed the scientific pool or ranking");
    }
    const auto detail_only = parse({"--pose-detail", "full", "--no-sdf"});
    require(detail_only.pose_output_controls && detail_only.pose_export.mode == PoseExportMode::TopK &&
            detail_only.pose_export.detail == PoseExportDetail::Full && detail_only.no_sdf,
            "Explicit full output or CSV-only compatibility differs");
    const auto legacy_top_k = parse({"--top-k", "3"});
    require(!legacy_top_k.pose_output_controls && legacy_top_k.pose_export.top_k == 3,
            "Legacy top-k unexpectedly enabled pool metrics");
}

void test_center_distance_options() {
    const auto defaults = parse();
    require(!defaults.pose_center_distance && defaults.center_reference.empty(),
            "Pose center distance became enabled by default");
    const auto off = parse({"--pose-center-distance", "off", "--center-reference-mol2", "missing/shared.mol2"});
    require(!off.pose_center_distance && off.center_reference == "missing/shared.mol2" &&
            !off.pose_output_controls && off.references.empty(),
            "Disabled centroid output changed another export or reference option");
    const auto shared = parse({"--pose-center-distance", "on", "--center-reference-mol2", "missing/shared.mol2",
        "--continue-on-error"});
    require(shared.pose_center_distance && shared.center_reference == "missing/shared.mol2" &&
            shared.references.empty() && shared.continue_on_error && !shared.pose_output_controls &&
            shared.top_k == defaults.top_k && !shared.settings.fast_prefilter,
            "Shared centroid reference changed ranking, pool or error recovery");
    const auto paired = parse({"--pose-center-distance", "on", "--reference-mol2", "paired.mol2"});
    require(paired.pose_center_distance && paired.center_reference.empty() && paired.references == "paired.mol2",
            "Center-distance output lost the existing paired reference fallback");
    const auto combined = parse({"--gpu-batch", "on", "--cpu-gpu-overlap", "on", "--kabsch", "quaternion",
        "--ranking", "vinardo", "--rescoring-backend", "cuda", "--timings", "--no-sdf",
        "--pose-center-distance", "on", "--center-reference-mol2", "shared.mol2"});
    require(combined.pose_center_distance && combined.gpu_batch && combined.cpu_gpu_overlap == true &&
            combined.references.empty(), "Shared centroid reference was rejected by batch/overlap");
    for (const auto& flags : std::vector<std::vector<std::string>>{
            {"--pose-center-distance"}, {"--pose-center-distance", "true"}, {"--pose-center-distance", "on"},
            {"--center-reference-mol2"}, {"--center-reference-mol2", ""},
            {"--pose-center-distance", "on", "--center-reference-mol2", "shared", "--pose-center-distance", "off"},
            {"--pose-center-distance", "on", "--reference-mol2", "paired", "--continue-on-error"},
            {"--pose-center-distance", "on", "--reference-mol2", "paired", "--cpu-gpu-overlap", "on",
             "--kabsch", "quaternion", "--timings", "--no-sdf"}})
        rejects([&] { parse(flags); });
}

void test_batch_options() {
    const auto defaults = parse();
    require(!defaults.gpu_batch && defaults.batch_size == 16 &&
            defaults.batch_memory_bytes == std::size_t{2048} * 1024 * 1024,
            "GPU batch changed its defaults or became enabled implicitly");
    const auto off = parse({"--gpu-batch", "off", "--ranking", "hybrid"});
    require(!off.gpu_batch && off.kabsch_method == "quaternion" &&
            off.settings.rescoring_backend == GpuRescoringBackend::Cpu,
            "Explicit batch OFF changed the default method or rescoring backend");
    const auto enabled = parse({"--gpu-batch", "on"});
    require(enabled.gpu_batch && enabled.batch_size == 16 && enabled.kabsch_method == "quaternion" &&
            enabled.query_method == "optimized" && !enabled.cpu_gpu_overlap.has_value(),
            "Batch ON did not accept the default quaternion selector");
    for (const char* size : {"1", "2", "4", "8", "64"}) {
        const auto selected = parse({"--gpu-batch", "on", "--batch-size", size,
            "--batch-memory-mib", "512", "--kabsch", "quaternion", "--timings",
            "--pose-output", "all", "--pose-detail", "full", "--reference-mol2", "reference"});
        require(selected.gpu_batch && selected.batch_size == std::stoull(size) &&
                selected.batch_memory_bytes == std::size_t{512} * 1024 * 1024 &&
                selected.settings.ranking.method == PoseScoreMethod::Fast &&
                selected.settings.rescoring_backend == GpuRescoringBackend::Cpu &&
                !selected.settings.fast_prefilter && selected.query_method == "optimized" &&
                selected.pose_export.mode == PoseExportMode::All && selected.references == "reference",
                "GPU batch changed the selected Fast pool or output controls");
    }
    require(parse({"--gpu-batch", "on", "--kabsch", "quaternion",
                   "--query-method", "optimized"}).query_method == "optimized",
            "Batch unnecessarily changed supported query preparation");
    for (const auto& flags : std::vector<std::vector<std::string>>{
            {"--gpu-batch", "auto"}, {"--gpu-batch", "on", "--kabsch", "svd"},
            {"--batch-size", "8"}, {"--batch-memory-mib", "512"},
            {"--gpu-batch", "off", "--batch-size", "4"},
            {"--gpu-batch", "on", "--kabsch", "quaternion", "--batch-size", "0"},
            {"--gpu-batch", "on", "--kabsch", "quaternion", "--batch-size", "65"},
            {"--gpu-batch", "on", "--kabsch", "quaternion", "--batch-memory-mib", "0"},
            {"--gpu-batch", "on", "--kabsch", "quaternion", "--batch-memory-mib", "18446744073709551615"},
            {"--gpu-batch", "on", "--kabsch", "quaternion", "--cpu-gpu-overlap", "off"},
            {"--gpu-batch", "on", "--kabsch", "quaternion", "--cpu-gpu-overlap", "on"},
            {"--gpu-batch", "on", "--kabsch", "quaternion", "--overlap-trace"},
            {"--gpu-batch", "on", "--kabsch", "quaternion", "--diagnostics", "stages"}}) {
        rejects([&] { parse(flags); });
    }
    for (const char* ranking : {"weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"}) {
        rejects([&] { parse({"--gpu-batch", "on", "--kabsch", "quaternion",
                            "--ranking", ranking}); });
        rejects([&] { parse({"--gpu-batch", "on", "--kabsch", "quaternion",
                            "--ranking", ranking, "--rescoring-backend", "cpu"}); });
        for (const char* size : {"1", "2", "4", "8"}) {
            for (const char* prefilter : {"none", "500"}) {
                std::vector<std::string> flags{"--gpu-batch", "on", "--batch-size", size,
                    "--kabsch", "quaternion", "--query-method", "reference",
                    "--ranking", ranking, "--rescoring-backend", "cuda",
                    "--fast-prefilter", prefilter};
                const std::string method = ranking;
                if (method == "weighted-interaction" || method == "hybrid")
                    flags.insert(flags.end(), {"--interaction-region", "protein"});
                if (method == "vinardo" || method == "hybrid")
                    flags.insert(flags.end(), {"--vinardo-alpha", "0.5"});
                if (method == "hybrid")
                    flags.insert(flags.end(), {"--hybrid-lambda", "0.6"});
                const auto selected = parse(flags);
                require(selected.gpu_batch && selected.batch_size == std::stoull(size) &&
                        selected.ranking_requested == ranking && selected.kabsch_method == "quaternion" &&
                        selected.query_method == "reference" &&
                        selected.settings.rescoring_backend == GpuRescoringBackend::Cuda &&
                        selected.settings.fast_prefilter == (std::string(prefilter) == "none"
                            ? std::nullopt : std::optional<std::size_t>{500}),
                        "Batch rescoring changed the selected protocol or candidate pool");
                const auto expected_method = method == "weighted-interaction"
                    ? PoseScoreMethod::WeightedInteraction : method == "hybrid"
                    ? PoseScoreMethod::Hybrid : PoseScoreMethod::Vinardo;
                require(selected.settings.ranking.method == expected_method &&
                        selected.settings.interaction_full_protein ==
                            (method == "weighted-interaction" || method == "hybrid") &&
                        selected.settings.ranking.vinardo_alpha ==
                            (method == "weighted-interaction" ? 1.0 : 0.5) &&
                        selected.settings.ranking.hybrid_lambda == (method == "hybrid" ? 0.6 : 0.3),
                        "Batch rescoring changed region, alpha or Hybrid lambda");
            }
        }
        for (const char* overlap : {"off", "on"})
            rejects([&] { parse({"--gpu-batch", "on", "--kabsch", "quaternion",
                "--ranking", ranking, "--rescoring-backend", "cuda", "--cpu-gpu-overlap", overlap}); });
    }
    for (const char* key : {"DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS"}) {
        environment(key, "0");
        rejects([] { parse({"--gpu-batch", "on", "--kabsch", "quaternion"}); });
        environment(key, std::nullopt);
    }
    for (const char* mode : {"off", "on"}) {
        for (const char* ranking : {"fast", "weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"}) {
            const auto selected = parse({"--gpu-batch", "on", "--batch-size", "16",
                "--cpu-gpu-overlap", mode, "--kabsch", "quaternion", "--query-method", "reference",
                "--ranking", ranking, "--rescoring-backend", "cuda", "--fast-prefilter", "none",
                "--timings", "--no-sdf", "--batch-memory-mib", "4096"});
            require(selected.gpu_batch && selected.batch_size == 16 &&
                    selected.cpu_gpu_overlap == (std::string(mode) == "on") &&
                    selected.ranking_requested == ranking && selected.kabsch_method == "quaternion" &&
                    !selected.settings.fast_prefilter && selected.query_method == "reference",
                    "Batch overlap changed the scientific protocol or selected mode");
        }
    }
    require(parse({"--gpu-batch", "on", "--cpu-gpu-overlap", "on", "--overlap-trace",
                   "--kabsch", "quaternion", "--timings", "--no-sdf"}).overlap_trace,
            "Batch overlap rejected its optional NVTX trace");
}

void test_overlap_options() {
    require(!parse().cpu_gpu_overlap.has_value() && !parse().overlap_trace,
            "Overlap or trace became enabled by default");
    for (const char* mode : {"off", "on"}) {
        const auto options = parse({"--cpu-gpu-overlap", mode, "--kabsch", "quaternion",
                                    "--timings", "--no-sdf", "--diagnostics", "off"});
        require(options.cpu_gpu_overlap.has_value() &&
                *options.cpu_gpu_overlap == (std::string(mode) == "on"),
                "Explicit overlap mode was lost");
        require(options.settings.ranking.method == PoseScoreMethod::Fast &&
                !options.settings.fast_prefilter && options.kabsch_method == "quaternion" &&
                options.query_method == "optimized", "Overlap changed scientific settings");
        require(options.settings.rescoring_backend == GpuRescoringBackend::Cpu,
                "Fast overlap changed the unused rescoring backend default");
        for (const char* ranking : {"weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"}) {
            std::vector<std::string> flags{
                "--cpu-gpu-overlap", mode, "--kabsch", "quaternion", "--timings",
                "--no-sdf", "--diagnostics", "off", "--ranking", ranking,
                "--fast-prefilter", "500"};
            rejects([&] { parse(flags); }); // Default CPU rescoring is not this experiment.
            auto cpu = flags;
            cpu.insert(cpu.end(), {"--rescoring-backend", "cpu"});
            rejects([&] { parse(cpu); });
            flags.insert(flags.end(), {"--rescoring-backend", "cuda"});
            const bool hybrid = std::string(ranking) == "hybrid";
            const bool interaction = hybrid || std::string(ranking) == "weighted-interaction";
            const bool vinardo = hybrid || std::string(ranking) == "vinardo";
            if (interaction) flags.insert(flags.end(), {"--interaction-region", "protein"});
            if (vinardo) flags.insert(flags.end(), {"--vinardo-alpha", "0.75"});
            if (hybrid) flags.insert(flags.end(), {"--hybrid-lambda", "0.6"});
            const auto rescoring = parse(flags);
            require(rescoring.cpu_gpu_overlap.has_value() &&
                    *rescoring.cpu_gpu_overlap == (std::string(mode) == "on") &&
                    rescoring.ranking_requested == ranking &&
                    rescoring.settings.rescoring_backend == GpuRescoringBackend::Cuda &&
                    rescoring.settings.fast_prefilter == 500 &&
                    rescoring.settings.interaction_full_protein == interaction,
                    "CUDA overlap changed the requested rescoring protocol");
            const auto expected_method = interaction
                ? (hybrid ? PoseScoreMethod::Hybrid : PoseScoreMethod::WeightedInteraction)
                : PoseScoreMethod::Vinardo;
            const double expected_alpha = vinardo ? 0.75
                : std::string(ranking) == "vinardo-rep50" ? 0.5 : 1.0;
            require(rescoring.settings.ranking.method == expected_method &&
                    rescoring.settings.ranking.vinardo_alpha == expected_alpha &&
                    rescoring.settings.ranking.hybrid_lambda == (hybrid ? 0.6 : 0.3),
                    "CUDA overlap changed scorer parameters or the rep50 convention");

            // Legacy calls retain CPU rescoring and do not enter the packet pipeline.
            const auto legacy = parse({"--ranking", ranking, "--rescoring-backend", "cpu"});
            require(!legacy.cpu_gpu_overlap.has_value() &&
                    legacy.settings.rescoring_backend == GpuRescoringBackend::Cpu,
                    "Overlap validation rejected or changed legacy CPU rescoring");
        }
        const std::vector<std::string> extended{
            "--cpu-gpu-overlap", mode, "--ranking", "hybrid", "--rescoring-backend", "cuda"};
        for (const auto& incompatible : std::vector<std::vector<std::string>>{
                {"--kabsch", "svd", "--timings", "--no-sdf"},
                {"--kabsch", "quaternion", "--no-sdf"}, // Missing timings.
                {"--kabsch", "quaternion", "--timings"}, // SDF still enabled.
                {"--kabsch", "quaternion", "--timings", "--no-sdf", "--diagnostics", "stages"},
                {"--kabsch", "quaternion", "--timings", "--no-sdf", "--reference-mol2", "reference"}}) {
            auto flags = extended;
            flags.insert(flags.end(), incompatible.begin(), incompatible.end());
            rejects([&] { parse(flags); });
        }
        auto no_poses = extended;
        no_poses.insert(no_poses.end(), {"--kabsch", "quaternion", "--timings", "--pose-output", "none"});
        require(parse(no_poses).pose_export.mode == PoseExportMode::None,
                "CUDA rescoring overlap rejected the no-pose alternative to --no-sdf");
        for (const char* key : {"DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS"}) {
            environment(key, "0");
            rejects([&] { parse(no_poses); });
            environment(key, std::nullopt);
        }
    }
    const auto trace = parse({"--cpu-gpu-overlap", "on", "--kabsch", "quaternion",
                              "--timings", "--no-sdf", "--overlap-trace"});
    require(trace.overlap_trace, "Explicit overlap tracing was lost");
    for (const auto& invalid : std::vector<std::vector<std::string>>{
            {"--cpu-gpu-overlap", "on"}, {"--cpu-gpu-overlap", "other"},
            {"--overlap-trace"},
            {"--cpu-gpu-overlap", "on", "--kabsch", "svd", "--timings", "--no-sdf"},
            {"--cpu-gpu-overlap", "on", "--kabsch", "quaternion", "--timings"},
            {"--cpu-gpu-overlap", "on", "--kabsch", "quaternion", "--timings", "--no-sdf", "--ranking", "hybrid"},
            {"--cpu-gpu-overlap", "on", "--kabsch", "quaternion", "--timings", "--no-sdf", "--diagnostics", "stages"},
            {"--cpu-gpu-overlap", "on", "--kabsch", "quaternion", "--timings", "--no-sdf", "--reference-mol2", "reference"}})
        rejects([&] { parse(invalid); });
    environment("DOCKING_KABSCH_TIMINGS", "0");
    rejects([] { parse({"--cpu-gpu-overlap", "off", "--kabsch", "quaternion", "--timings", "--no-sdf"}); });
    environment("DOCKING_KABSCH_TIMINGS", std::nullopt);
}

void test_ranking_compatibility() {
    for (const char* ranking : {"fast", "weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"}) {
        const auto options = parse({"--ranking", ranking, "--fast-prefilter", "none"});
        require(options.ranking_requested == ranking && !options.settings.fast_prefilter,
                "Ranking request or explicit none prefilter differs");
    }
    for (const char* ranking : {"weighted-interaction", "hybrid"})
        for (const char* region : {"pocket", "protein"})
            require(parse({"--ranking", ranking, "--interaction-region", region}).settings.interaction_full_protein ==
                    (std::string(region) == "protein"), "WeightedInteraction-region selection differs");
    for (const char* ranking : {"fast", "vinardo", "vinardo-rep50"})
        for (const char* region : {"pocket", "protein"})
            rejects([&] { parse({"--ranking", ranking, "--interaction-region", region}); });
    rejects([] { parse({"--fast-prefilter", "10"}); });
    for (const char* ranking : {"weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"})
        require(parse({"--ranking", ranking, "--fast-prefilter", "10"}).settings.fast_prefilter == 10,
                "Rescoring Fast prefilter rejected");
    for (const char* ranking : {"fast", "weighted-interaction"})
        rejects([&] { parse({"--ranking", ranking, "--vinardo-alpha", "1"}); });
    for (const char* ranking : {"vinardo", "hybrid"}) {
        require(parse({"--ranking", ranking, "--vinardo-alpha", "0"}).settings.ranking.vinardo_alpha == 0,
                "Zero Vinardo alpha rejected");
        require(parse({"--ranking", ranking, "--vinardo-alpha", "2"}).settings.ranking.vinardo_alpha == 2,
                "Vinardo alpha above one rejected");
    }
    const auto alias = parse({"--ranking", "vinardo-rep50"});
    require(alias.settings.ranking.method == PoseScoreMethod::Vinardo && alias.settings.ranking.vinardo_alpha == .5,
            "rep50 alias did not set Vinardo alpha");
    require(parse({"--vinardo-alpha", "0.5", "--ranking", "vinardo-rep50"}).settings.ranking.vinardo_alpha == .5,
            "Compatible alias parameter rejected or option order changed meaning");
    rejects([] { parse({"--ranking", "vinardo-rep50", "--vinardo-alpha", "1"}); });
    rejects([] { parse({"--vinardo-alpha", "0", "--ranking", "vinardo-rep50"}); });
    for (const char* ranking : {"fast", "weighted-interaction", "vinardo", "vinardo-rep50"})
        rejects([&] { parse({"--ranking", ranking, "--hybrid-lambda", ".3"}); });
    for (const char* endpoint : {"0", "1"})
        require(parse({"--ranking", "hybrid", "--hybrid-lambda", endpoint}).settings.ranking.hybrid_lambda ==
                std::stod(endpoint), "Hybrid lambda endpoint rejected");
}

void test_rescoring_backend() {
    for (const char* method : {"weighted-interaction", "vinardo", "vinardo-rep50", "hybrid"}) {
        require(parse({"--ranking", method}).settings.rescoring_backend == GpuRescoringBackend::Cpu,
                "Rescoring default no longer preserves the CPU reference");
        require(parse({"--ranking", method, "--rescoring-backend", "cpu"}).settings.rescoring_backend ==
                    GpuRescoringBackend::Cpu, "Explicit CPU rescoring was not selected");
        require(parse({"--rescoring-backend", "cuda", "--ranking", method}).settings.rescoring_backend ==
                    GpuRescoringBackend::Cuda, "CUDA rescoring or CLI option order differs");
    }
    const auto hybrid = parse({"--ranking", "hybrid", "--rescoring-backend", "cuda",
                               "--vinardo-alpha", ".5", "--hybrid-lambda", ".3"});
    require(hybrid.settings.ranking.vinardo_alpha == .5 && hybrid.settings.ranking.hybrid_lambda == .3 &&
            !hybrid.settings.fast_prefilter, "CUDA backend changed the Hybrid scientific settings");
}

void test_invalid_arguments() {
    const std::vector<std::vector<std::string>> invalid = {
        {"--unknown"}, {"extra-positional"}, {"--top-k"}, {"--reference-mol2"}, {"--reference-mol2", ""},
        {"--top-k", "0"}, {"--top-k", "-1"}, {"--top-k", "1.5"}, {"--top-k", " 1"},
        {"--top-k", "999999999999999999999999999999999999"}, {"--limit", "0"}, {"--limit", "+1"},
        {"--cutoff", "0"}, {"--cutoff", "-1"}, {"--cutoff", "1x"}, {"--cutoff", " 2"},
        {"--cutoff", "nan"}, {"--cutoff", "inf"}, {"--cutoff", "1e9999"}, {"--ranking", "unknown"},
        {"--ranking", "exact"}, {"--ranking", "hybrid", "--exact-region", "pocket"},
        {"--kabsch", "SVD"}, {"--query-method", "fast"}, {"--clash-threshold", "1.01"},
        {"--clash-threshold", "-0.01"}, {"--clash-threshold", "NaN"}, {"--min-target-clearance", "-1"},
        {"--min-target-clearance", "inf"}, {"--min-target-clearance", "OFF"}, {"--interaction-region", "all"},
        {"--ranking", "weighted-interaction", "--fast-prefilter", "0"}, {"--ranking", "weighted-interaction", "--fast-prefilter", "-2"},
        {"--ranking", "vinardo", "--vinardo-alpha", "-1"}, {"--ranking", "hybrid", "--vinardo-alpha", "nan"},
        {"--ranking", "hybrid", "--hybrid-lambda", "1.01"}, {"--ranking", "hybrid", "--hybrid-lambda", "-1"},
        {"--ranking", "hybrid", "--hybrid-lambda", "inf"}, {"--diagnostics", "1"},
        {"--top-k", "2", "--top-k", "3"}, {"--timings", "--timings"},
        {"--continue-on-error", "--continue-on-error"}, {"--continue-on-error", "true"},
        {"--no-sdf", "--no-sdf"}, {"--no-sdf", "true"},
        {"--continue-on-error", "--reference-mol2", "missing/references.mol2"},
        {"--reference-mol2", "missing/references.mol2", "--continue-on-error"},
        {"--min-target-clearance", "off", "--min-target-clearance", "0"},
        {"--rescoring-backend"}, {"--rescoring-backend", ""},
        {"--rescoring-backend", "gpu"}, {"--rescoring-backend", "CUDA"},
        {"--rescoring-backend", " cuda"},
        {"--rescoring-backend", "cpu", "--rescoring-backend", "cuda"},
        {"--rescore-backend", "cuda"},
        {"--pose-output"}, {"--pose-output", "None"}, {"--pose-output", "invalid"},
        {"--pose-detail"}, {"--pose-detail", "small"}, {"--pose-detail", "FULL"},
        {"--pose-output", "none", "--top-k", "0"},
        {"--pose-output", "all", "--pose-output", "none"},
        {"--pose-detail", "compact", "--pose-detail", "full"}
    };
    for (const auto& flags : invalid) rejects([&] { parse(flags); });
    rejects([] { parse_gpu_docking_options(0, nullptr); });
    std::vector<std::string> few = {"gpu_docking", "protein", "pocket", "ligands"};
    std::vector<char*> argv;
    for (auto& argument : few) argv.push_back(argument.data());
    rejects([&] { parse_gpu_docking_options(static_cast<int>(argv.size()), argv.data()); });
}

void test_environment_precedence() {
    environment("DOCKING_KABSCH_METHOD", "svd");
    environment("DOCKING_QUERY_METHOD", "reference");
    environment("DOCKING_PROFILE_STAGES", "1");
    const auto inherited = parse();
    require(inherited.kabsch_method == "svd" && inherited.query_method == "reference" &&
            inherited.settings.diagnostic_stages && inherited.kabsch_source == "environment" &&
            inherited.query_source == "environment" && inherited.diagnostics_source == "environment",
            "Environment values or sources differ");
    const auto overridden = parse({"--kabsch", "quaternion", "--query-method", "optimized", "--diagnostics", "off"});
    require(overridden.kabsch_method == "quaternion" && overridden.query_method == "optimized" &&
            !overridden.settings.diagnostic_stages && overridden.kabsch_source == "cli" &&
            overridden.query_source == "cli" && overridden.diagnostics_source == "cli",
            "CLI failed to override environment");
    require(environment("DOCKING_KABSCH_METHOD") == "svd" &&
            environment("DOCKING_QUERY_METHOD") == "reference" &&
            environment("DOCKING_PROFILE_STAGES") == "1", "Parser changed inherited selectors");
    environment("DOCKING_KABSCH_METHOD", std::nullopt);
    environment("DOCKING_QUERY_METHOD", std::nullopt);
    for (const char* value : {"0", "true", "", "01", "1 "}) {
        environment("DOCKING_PROFILE_STAGES", value);
        const auto disabled = parse();
        require(!disabled.settings.diagnostic_stages && disabled.diagnostics_source == "environment",
                "Diagnostics did not preserve exact-1 rule");
        require(parse({"--diagnostics", "stages"}).settings.diagnostic_stages,
                "CLI could not enable diagnostics over non-1 environment");
    }
    for (const char* value : {"", "auto", "SVD"}) {
        environment("DOCKING_KABSCH_METHOD", value);
        rejects([] { parse(); });
        require(parse({"--kabsch", "svd"}).kabsch_source == "cli", "Valid CLI did not mask invalid Kabsch env");
    }
    environment("DOCKING_KABSCH_METHOD", std::nullopt);
    for (const char* value : {"", "auto", "OPTIMIZED"}) {
        environment("DOCKING_QUERY_METHOD", value);
        rejects([] { parse(); });
        require(parse({"--query-method", "reference"}).query_source == "cli", "Valid CLI did not mask invalid query env");
    }
    environment("DOCKING_QUERY_METHOD", "invalid");
    environment("DOCKING_KABSCH_METHOD", "invalid");
    environment("DOCKING_PROFILE_STAGES", "1");
    environment("DOCKING_STAGE_TIMINGS", "preserve-me");
    const auto effective = parse({"--kabsch", "quaternion", "--query-method", "optimized", "--diagnostics", "off"});
    apply_gpu_docking_environment(effective);
    require(environment("DOCKING_KABSCH_METHOD") == "quaternion" && environment("DOCKING_QUERY_METHOD") == "optimized",
            "Effective methods were not applied");
    require(environment("DOCKING_PROFILE_STAGES") == "1" && environment("DOCKING_STAGE_TIMINGS") == "preserve-me",
            "Environment application changed unrelated or diagnostic selectors");
    auto invalid_effective = effective;
    invalid_effective.query_method = "invalid";
    rejects([&] { apply_gpu_docking_environment(invalid_effective); });
}

} // namespace

int main() {
    try {
        EnvironmentGuard guard;
        test_defaults_and_values();
        test_pose_output_options();
        test_center_distance_options();
        test_batch_options();
        test_overlap_options();
        test_ranking_compatibility();
        test_rescoring_backend();
        test_invalid_arguments();
        test_environment_precedence();
        std::ostringstream usage;
        write_gpu_docking_usage(usage);
        require(usage.str().find("vinardo-rep50") != std::string::npos &&
                usage.str().find("--min-target-clearance") != std::string::npos &&
                usage.str().find("--rescoring-backend") != std::string::npos &&
                usage.str().find("--continue-on-error") != std::string::npos &&
                usage.str().find("--no-sdf") != std::string::npos &&
                usage.str().find("--pose-output") != std::string::npos &&
                usage.str().find("--pose-detail") != std::string::npos &&
                usage.str().find("--gpu-batch") != std::string::npos &&
                usage.str().find("--batch-size") != std::string::npos &&
                usage.str().find("--batch-memory-mib") != std::string::npos &&
                usage.str().find("--pose-center-distance") != std::string::npos &&
                usage.str().find("--center-reference-mol2") != std::string::npos,
                "Usage omits supported settings");
        std::cout << "GPU docking configuration tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "GPU docking configuration test failed: " << exception.what() << '\n';
        return 1;
    }
}
