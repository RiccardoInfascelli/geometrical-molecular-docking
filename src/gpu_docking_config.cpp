#include "gpu_docking_config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

std::size_t positive_integer(const std::string& value, const std::string& flag) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(flag + " requires a positive integer");
    try {
        const auto number = std::stoull(value);
        if (!number || number > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(flag + " is outside the supported range");
        return static_cast<std::size_t>(number);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(flag + " is outside the supported range");
    }
}

double finite_number(const std::string& value, const std::string& flag) {
    if (value.empty() || value.find_first_of(" \t\n\r\f\v") != std::string::npos)
        throw std::invalid_argument(flag + " requires a finite number");
    try {
        std::size_t parsed = 0;
        const double number = std::stod(value, &parsed);
        if (parsed != value.size() || !std::isfinite(number))
            throw std::invalid_argument(flag + " requires a finite number");
        return number;
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(flag + " requires a finite number");
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(flag + " requires a finite number in range");
    }
}

void validate_kabsch(const std::string& value, const std::string& source) {
    if (value != "svd" && value != "quaternion")
        throw std::invalid_argument(source + " must be svd or quaternion");
}

void validate_query(const std::string& value, const std::string& source) {
    if (value != "reference" && value != "optimized")
        throw std::invalid_argument(source + " must be reference or optimized");
}

std::string positional(int argc, char** argv, int index) {
    if (index >= argc || !argv || !argv[index] || !*argv[index] ||
        std::string(argv[index]).starts_with("--"))
        throw std::invalid_argument("Expected protein, pocket, ligand MOL2 and output paths before options");
    return argv[index];
}

} // namespace

GpuDockingCliOptions parse_gpu_docking_options(int argc, char** argv) {
    if (argc < 5 || !argv)
        throw std::invalid_argument("gpu_docking requires protein, pocket, ligand MOL2 and output paths");
    GpuDockingCliOptions result;
    result.config.protein_path = positional(argc, argv, 1);
    result.config.pocket_path = positional(argc, argv, 2);
    result.input = positional(argc, argv, 3);
    result.output = positional(argc, argv, 4);
    result.config.ligand_path = result.input;
    result.config.output_directory = result.output;
    result.config.output_mode = OutputMode::Quiet;
    std::unordered_set<std::string> seen;
    bool interaction_region_given = false, alpha_given = false, lambda_given = false;
    for (int i = 5; i < argc; ++i) {
        if (!argv[i]) throw std::invalid_argument("Null command-line argument");
        const std::string flag = argv[i];
        if (!seen.insert(flag).second)
            throw std::invalid_argument("Duplicate option: " + flag);
        if (flag == "--timings") { result.settings.collect_timings = true; continue; }
        if (flag != "--top-k" && flag != "--limit" && flag != "--cutoff" &&
            flag != "--reference-mol2" && flag != "--ranking" && flag != "--kabsch" &&
            flag != "--query-method" && flag != "--clash-threshold" &&
            flag != "--min-target-clearance" && flag != "--interaction-region" &&
            flag != "--fast-prefilter" && flag != "--vinardo-alpha" &&
            flag != "--hybrid-lambda" && flag != "--diagnostics" && flag != "--rescoring-backend")
            throw std::invalid_argument("Unknown option: " + flag);
        if (++i >= argc || !argv[i] || !*argv[i] || std::string(argv[i]).starts_with("--"))
            throw std::invalid_argument("Missing value for " + flag);
        const std::string value = argv[i];
        if (flag == "--top-k") result.top_k = positive_integer(value, flag);
        else if (flag == "--limit") result.limit = positive_integer(value, flag);
        else if (flag == "--reference-mol2") result.references = value;
        else if (flag == "--cutoff") {
            result.settings.cutoff = finite_number(value, flag);
            if (result.settings.cutoff <= 0.0)
                throw std::invalid_argument("--cutoff must be positive");
        } else if (flag == "--ranking") {
            result.ranking_requested = value;
            if (value == "fast") result.settings.ranking.method = PoseScoreMethod::Fast;
            else if (value == "weighted-interaction") result.settings.ranking.method = PoseScoreMethod::WeightedInteraction;
            else if (value == "vinardo" || value == "vinardo-rep50")
                result.settings.ranking.method = PoseScoreMethod::Vinardo;
            else if (value == "hybrid") result.settings.ranking.method = PoseScoreMethod::Hybrid;
            else throw std::invalid_argument("--ranking must be fast, weighted-interaction, vinardo, vinardo-rep50 or hybrid");
        } else if (flag == "--kabsch") {
            validate_kabsch(value, flag);
            result.kabsch_method = value;
            result.kabsch_source = "cli";
        } else if (flag == "--query-method") {
            validate_query(value, flag);
            result.query_method = value;
            result.query_source = "cli";
        } else if (flag == "--clash-threshold") {
            result.config.operational_clash_threshold = finite_number(value, flag);
            if (result.config.operational_clash_threshold < 0.0 ||
                result.config.operational_clash_threshold > 1.0)
                throw std::invalid_argument("--clash-threshold must be in [0, 1]");
        } else if (flag == "--min-target-clearance") {
            if (value == "off") result.settings.min_target_clearance.reset();
            else {
                const double minimum = finite_number(value, flag);
                if (minimum < 0.0)
                    throw std::invalid_argument("--min-target-clearance must be off or non-negative");
                result.settings.min_target_clearance = minimum;
            }
        } else if (flag == "--interaction-region") {
            interaction_region_given = true;
            if (value != "pocket" && value != "protein")
                throw std::invalid_argument("--interaction-region must be pocket or protein");
            result.settings.interaction_full_protein = value == "protein";
        } else if (flag == "--fast-prefilter") {
            if (value == "none") result.settings.fast_prefilter.reset();
            else result.settings.fast_prefilter = positive_integer(value, flag);
        } else if (flag == "--vinardo-alpha") {
            alpha_given = true;
            result.settings.ranking.vinardo_alpha = finite_number(value, flag);
            if (result.settings.ranking.vinardo_alpha < 0.0)
                throw std::invalid_argument("--vinardo-alpha must be non-negative");
        } else if (flag == "--hybrid-lambda") {
            lambda_given = true;
            result.settings.ranking.hybrid_lambda = finite_number(value, flag);
            if (result.settings.ranking.hybrid_lambda < 0.0 ||
                result.settings.ranking.hybrid_lambda > 1.0)
                throw std::invalid_argument("--hybrid-lambda must be in [0, 1]");
        } else if (flag == "--rescoring-backend") {
            if (value == "cpu") result.settings.rescoring_backend = GpuRescoringBackend::Cpu;
            else if (value == "cuda") result.settings.rescoring_backend = GpuRescoringBackend::Cuda;
            else throw std::invalid_argument("--rescoring-backend must be cpu or cuda");
        } else if (flag == "--diagnostics") {
            if (value != "off" && value != "stages")
                throw std::invalid_argument("--diagnostics must be off or stages");
            result.settings.diagnostic_stages = value == "stages";
            result.diagnostics_source = "cli";
        }
    }
    // Explicit scorer options are rejected for unrelated methods, even at default values.
    const auto method = result.settings.ranking.method;
    if (method == PoseScoreMethod::Fast && result.settings.fast_prefilter)
        throw std::invalid_argument("--fast-prefilter requires a rescoring ranking");
    if (interaction_region_given && method != PoseScoreMethod::WeightedInteraction && method != PoseScoreMethod::Hybrid)
        throw std::invalid_argument("--interaction-region requires weighted-interaction or hybrid ranking");
    if (alpha_given && method != PoseScoreMethod::Vinardo && method != PoseScoreMethod::Hybrid)
        throw std::invalid_argument("--vinardo-alpha requires vinardo, vinardo-rep50 or hybrid ranking");
    if (lambda_given && method != PoseScoreMethod::Hybrid)
        throw std::invalid_argument("--hybrid-lambda requires hybrid ranking");
    if (result.ranking_requested == "vinardo-rep50") {
        if (alpha_given && result.settings.ranking.vinardo_alpha != 0.5)
            throw std::invalid_argument("vinardo-rep50 fixes --vinardo-alpha to 0.5");
        result.settings.ranking.vinardo_alpha = 0.5;
    }
    // Consult environment selectors only when no CLI value was supplied.
    if (result.kabsch_source.empty()) {
        const char* environment = std::getenv("DOCKING_KABSCH_METHOD");
        result.kabsch_method = environment ? environment : "svd";
        result.kabsch_source = environment ? "environment" : "default";
        validate_kabsch(result.kabsch_method, "DOCKING_KABSCH_METHOD");
    }
    if (result.query_source.empty()) {
        const char* environment = std::getenv("DOCKING_QUERY_METHOD");
        result.query_method = environment ? environment : "reference";
        result.query_source = environment ? "environment" : "default";
        validate_query(result.query_method, "DOCKING_QUERY_METHOD");
    }
    if (result.diagnostics_source.empty()) {
        const char* environment = std::getenv("DOCKING_PROFILE_STAGES");
        result.settings.diagnostic_stages = environment && std::strcmp(environment, "1") == 0;
        result.diagnostics_source = environment ? "environment" : "default";
    }
    return result;
}

void apply_gpu_docking_environment(const GpuDockingCliOptions& options) {
    validate_kabsch(options.kabsch_method, "Effective Kabsch method");
    validate_query(options.query_method, "Effective query method");
    if (::setenv("DOCKING_KABSCH_METHOD", options.kabsch_method.c_str(), 1) != 0)
        throw std::runtime_error(std::string("Could not set DOCKING_KABSCH_METHOD: ") + std::strerror(errno));
    if (::setenv("DOCKING_QUERY_METHOD", options.query_method.c_str(), 1) != 0)
        throw std::runtime_error(std::string("Could not set DOCKING_QUERY_METHOD: ") + std::strerror(errno));
}

void write_gpu_docking_usage(std::ostream& output) {
    output << "Usage: gpu_docking <protein.pdb> <pocket.pdb> <ligands.mol2> <new-output-directory>\n"
        "  --top-k K                     Poses per ligand (default 20)\n"
        "  --limit N                     First N records (default all)\n"
        "  --cutoff A                    Shared triplet cutoff, positive (default 24)\n"
        "  --ranking METHOD              fast|weighted-interaction|vinardo|vinardo-rep50|hybrid (default fast)\n"
        "  --kabsch METHOD               svd|quaternion (default svd)\n"
        "  --query-method METHOD         reference|optimized (default reference)\n"
        "  --clash-threshold FRACTION    Fraction in [0,1] (default 0.05)\n"
        "  --min-target-clearance off|A  Optional non-negative target filter (default off; 0 is active)\n"
        "  --interaction-region pocket|protein WeightedInteraction/Hybrid target region (default pocket)\n"
        "  --fast-prefilter none|N       Positive Fast pool limit before rescoring (default none)\n"
        "  --vinardo-alpha VALUE         Non-negative repulsion weight (default 1; rep50 fixes 0.5)\n"
        "  --hybrid-lambda FRACTION       Hybrid WeightedInteraction weight in [0,1] (default 0.3)\n"
        "  --rescoring-backend cpu|cuda   WeightedInteraction/Vinardo/Hybrid computation (default cpu)\n"
        "  --timings                     Write completed host wall intervals\n"
        "  --diagnostics off|stages      Stage diagnostics (default off)\n"
        "  --reference-mol2 FILE         Optional records paired by position and atom order\n"
        "CLI selectors override DOCKING_KABSCH_METHOD, DOCKING_QUERY_METHOD and\n"
        "DOCKING_PROFILE_STAGES. Only the exact environment value 1 enables stages.\n"
        "Kabsch/query environment values must be canonical unless overridden by CLI;\n"
        "unrecognized Kabsch values are errors here, unlike the legacy SVD fallback.\n"
        "Protein, pocket, ligand and references must use the same coordinate frame.\n"
        "RMSD never participates in ranking; it is calculated only for exported poses.\n";
}
