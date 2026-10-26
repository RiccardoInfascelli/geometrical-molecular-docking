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
                                 "DOCKING_PROFILE_STAGES", "DOCKING_STAGE_TIMINGS"}) {
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
    require(defaults.settings.cutoff == 24.0 && defaults.ranking_requested == "fast" &&
            defaults.settings.ranking.method == PoseScoreMethod::Fast &&
            defaults.settings.rescoring_backend == GpuRescoringBackend::Cpu &&
            !defaults.settings.collect_timings && !defaults.settings.diagnostic_stages &&
            !defaults.settings.min_target_clearance && !defaults.settings.fast_prefilter,
            "Default scientific settings differ");
    require(defaults.kabsch_method == "svd" && defaults.query_method == "reference" &&
            defaults.kabsch_source == "default" && defaults.query_source == "default" &&
            defaults.diagnostics_source == "default", "Default selectors/sources differ");
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
        {"--min-target-clearance", "off", "--min-target-clearance", "0"},
        {"--rescoring-backend"}, {"--rescoring-backend", ""},
        {"--rescoring-backend", "gpu"}, {"--rescoring-backend", "CUDA"},
        {"--rescoring-backend", " cuda"},
        {"--rescoring-backend", "cpu", "--rescoring-backend", "cuda"},
        {"--rescore-backend", "cuda"}
    };
    for (const auto& flags : invalid) rejects([&] { parse(flags); });
    rejects([] { parse_gpu_docking_options(0, nullptr); });
    std::vector<std::string> few = {"gpu_docking", "protein", "pocket", "ligands"};
    std::vector<char*> argv;
    for (auto& argument : few) argv.push_back(argument.data());
    rejects([&] { parse_gpu_docking_options(static_cast<int>(argv.size()), argv.data()); });
}

void test_environment_precedence() {
    environment("DOCKING_KABSCH_METHOD", "quaternion");
    environment("DOCKING_QUERY_METHOD", "optimized");
    environment("DOCKING_PROFILE_STAGES", "1");
    const auto inherited = parse();
    require(inherited.kabsch_method == "quaternion" && inherited.query_method == "optimized" &&
            inherited.settings.diagnostic_stages && inherited.kabsch_source == "environment" &&
            inherited.query_source == "environment" && inherited.diagnostics_source == "environment",
            "Environment values or sources differ");
    const auto overridden = parse({"--kabsch", "svd", "--query-method", "reference", "--diagnostics", "off"});
    require(overridden.kabsch_method == "svd" && overridden.query_method == "reference" &&
            !overridden.settings.diagnostic_stages && overridden.kabsch_source == "cli" &&
            overridden.query_source == "cli" && overridden.diagnostics_source == "cli",
            "CLI failed to override environment");
    require(environment("DOCKING_KABSCH_METHOD") == "quaternion" &&
            environment("DOCKING_QUERY_METHOD") == "optimized" &&
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
        test_ranking_compatibility();
        test_rescoring_backend();
        test_invalid_arguments();
        test_environment_precedence();
        std::ostringstream usage;
        write_gpu_docking_usage(usage);
        require(usage.str().find("vinardo-rep50") != std::string::npos &&
                usage.str().find("--min-target-clearance") != std::string::npos &&
                usage.str().find("--rescoring-backend") != std::string::npos,
                "Usage omits supported settings");
        std::cout << "GPU docking configuration tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "GPU docking configuration test failed: " << exception.what() << '\n';
        return 1;
    }
}
