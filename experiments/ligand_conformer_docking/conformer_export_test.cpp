// Exercise the actual output helpers on the host, without invoking the CLI.
#define CONFORMER_EXPORT_TEST
#define main conformer_export_app_main
#include "ligand_conformer_docking.cpp"
#undef main

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <iterator>
#include <unistd.h>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

RDKit::RWMol one_atom() {
    RDKit::RWMol molecule;
    molecule.addAtom(new RDKit::Atom(6), true, true);
    auto conformer = std::make_unique<RDKit::Conformer>(1);
    conformer->setAtomPos(0, RDGeom::Point3D(0, 0, 0));
    molecule.addConformer(conformer.release(), true);
    return molecule;
}

Record synthetic_record(std::size_t id) {
    Record record;
    record.id = id;
    record.molecule = one_atom();
    for (std::size_t i = 0; i < 25; ++i) {
        EnsemblePose row;
        row.conformer_id = id;
        row.pose_id = row.scored.pose_id = i;
        row.alignment_index = i + 100;
        auto& pose = row.scored.pose;
        pose.fast_weighted_score = pose.weighted_interaction_score.weighted_score = 25.0 - i;
        row.scored.vinardo.gaussian = row.scored.vinardo.total = -(25.0 - i);
        pose.transform.translation[0] = i == 24 ? 0.1 : 3.0 + i * 0.01;
        GpuRigidTransform transform{};
        std::copy(pose.transform.rotation.begin(), pose.transform.rotation.end(), transform.rotation);
        std::copy(pose.transform.translation.begin(), pose.transform.translation.end(), transform.translation);
        transform.valid = 1;
        row.rmsd = gpu_pose_heavy_atom_rmsd(record.molecule, transform, record.molecule);
        record.docking.poses.push_back(row);
        record.docking.rescored_ids.push_back(i);
    }
    record.docking.survivors = record.docking.poses.size();
    record.docking.oracle_all_survivors = record.docking.oracle_after_prefilter = 0.1;
    return record;
}

std::string contents(const fs::path& path) {
    std::ifstream input(path);
    check(bool(input), "Missing synthetic output");
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::vector<std::vector<std::string>> table(const fs::path& path) {
    std::ifstream input(path);
    check(bool(input), "Missing synthetic table");
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(input, line)) {
        std::vector<std::string> fields;
        std::istringstream row(line);
        std::string field;
        while (std::getline(row, field, ',')) fields.push_back(field);
        if (!line.empty() && line.back() == ',') fields.emplace_back();
        rows.push_back(std::move(fields));
    }
    return rows;
}

void test_outputs(const fs::path& directory) {
    const std::vector<Record> records{synthetic_record(0), synthetic_record(1)};
    const auto rankings = rank_populations(records);
    check(rankings.size() == 8, "Expected two populations and four methods");
    for (const auto& result : rankings) {
        check(result.pool.size() == 25 && result.ranked.size() == 25, "Fixture pool must be complete");
        check(result.summary.oracle_rmsd && std::abs(*result.summary.oracle_rmsd - 0.1) < 1e-12,
              "Oracle must include the good pose outside the exported top-k");
        check(result.summary.top_k_best_rmsd.back() && *result.summary.top_k_best_rmsd.back() > 2.0,
              "Fixture good pose must remain outside Top-20 for every ranking");
    }
    ConformerEnsemble ensemble;
    ensemble.requested = ensemble.embedded = ensemble.distinct = 1;
    const auto summary = [&](const fs::path& path) {
        write_summary(path, rankings, ensemble, records, 1.0, 2.0);
    };
    summary(directory / "reference-summary.csv");
    const auto expected_summary = contents(directory / "reference-summary.csv");
    for (auto mode : {PoseExportMode::None, PoseExportMode::TopK, PoseExportMode::All}) {
        for (auto detail : {PoseExportDetail::Compact, PoseExportDetail::Full}) {
            const auto name = std::string(pose_export_mode_name(mode)) +
                (detail == PoseExportDetail::Compact ? "-compact" : "-full");
            const auto output = directory / name;
            fs::create_directory(output);
            Options options;
            options.pose_output_explicit = true;
            options.pose_export = {mode, 3, detail};
            options.sdf = mode == PoseExportMode::None; // none must suppress requested SDF too.
            write_rankings(output, records, rankings, options);
            summary(output / "summary.csv");
            check(contents(output / "summary.csv") == expected_summary,
                  "Export policy must not alter summary fields, RMSD, oracle or full-pool counts");
            if (mode == PoseExportMode::None) {
                check(!fs::exists(output / "poses.csv") && !fs::exists(output / "structures"),
                      "none must not create pose CSV or structure output");
                continue;
            }
            const auto exported = table(output / "poses.csv");
            const auto expected_rows = mode == PoseExportMode::TopK ? 8 * 3 : 8 * 25;
            check(exported.size() == std::size_t(expected_rows + 1), "Wrong serialized pose count");
            const auto columns = detail == PoseExportDetail::Compact ? 16 : 28;
            check(exported.front().size() == std::size_t(columns), "Wrong compact/full column count");
            for (std::size_t i = 1; i < exported.size(); ++i) {
                const auto& row = exported[i];
                check(row.size() == std::size_t(columns), "Inconsistent CSV row shape");
                const auto rank = std::stoul(row[6]);
                check(std::stoul(row[3]) == rank - 1, "Export changed pose identity or rank order");
                const auto expected_rmsd = rank == 25 ? 0.1 : 3.0 + (rank - 1) * 0.01;
                check(std::abs(std::stod(row[12]) - expected_rmsd) < 1e-12,
                      "Export changed a pose RMSD");
                if (detail == PoseExportDetail::Full)
                    check(std::abs(std::stod(row[25]) - expected_rmsd) < 1e-12,
                          "Full export must retain the transform for offline replay");
            }
        }
    }
    const auto output = directory / "legacy";
    fs::create_directory(output);
    write_rankings(output, records, rankings, Options{});
    check(contents(output / "poses.csv") == contents(directory / "all-full/poses.csv"),
          "Default pose export must retain the historical full/all CSV");
}

struct QueryEnvironment {
    std::optional<std::string> saved;
    explicit QueryEnvironment(const std::optional<std::string>& value) {
        if (const char* current = std::getenv("DOCKING_QUERY_METHOD")) saved = current;
        check((value ? ::setenv("DOCKING_QUERY_METHOD", value->c_str(), 1)
                     : ::unsetenv("DOCKING_QUERY_METHOD")) == 0, "Cannot isolate query-method environment");
    }
    ~QueryEnvironment() {
        if (saved) ::setenv("DOCKING_QUERY_METHOD", saved->c_str(), 1);
        else ::unsetenv("DOCKING_QUERY_METHOD");
    }
};

Options cli_options(const std::vector<std::string>& flags = {},
                    const std::optional<std::string>& query_environment = std::nullopt) {
    const QueryEnvironment environment(query_environment);
    std::vector<std::string> arguments{"conformer-test", "missing-complex", "new-output", "--generate-only"};
    arguments.insert(arguments.end(), flags.begin(), flags.end());
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return parse(static_cast<int>(argv.size()), argv.data());
}

Options diagnostic_cli(const std::vector<std::string>& flags = {}) {
    const QueryEnvironment environment(std::nullopt);
    std::vector<std::string> arguments{"conformer-test", "missing-complex", "new-output",
        "--diagnostics", "--rescoring-backend", "cuda"};
    arguments.insert(arguments.end(), flags.begin(), flags.end());
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return parse(static_cast<int>(argv.size()), argv.data());
}

void test_cli_protocols() {
    const auto defaults = cli_options();
    check(!defaults.execution.gpu_batch && !defaults.execution.cpu_gpu_overlap.has_value() &&
          defaults.execution.rescoring_backend == GpuRescoringBackend::Cpu &&
          defaults.execution.query_method == "optimized" && defaults.execution.batch_size == 4 &&
          defaults.execution.batch_memory_bytes == std::size_t{2048} * 1024 * 1024 &&
          !defaults.rmsd_oracle && !defaults.prefilter &&
          !defaults.pose_center_distance,
          "Unexpected conformer CLI defaults or scientific protocol");
    const ConformerExecutionOptions api_defaults;
    check(api_defaults.query_method == "reference" && api_defaults.batch_size == 8,
          "CLI defaults changed the reusable execution API");
    const auto batch_defaults = cli_options({"--gpu-batch", "on", "--rescoring-backend", "cuda"});
    check(batch_defaults.execution.gpu_batch && batch_defaults.execution.batch_size == 4 &&
          !batch_defaults.execution.cpu_gpu_overlap.has_value(), "Batch ON did not retain size 4 and overlap OFF");
    check(cli_options({"--gpu-batch", "on", "--rescoring-backend", "cuda", "--batch-size", "8"})
              .execution.batch_size == 8, "Explicit conformer batch size was ignored");
    check(cli_options({}, "reference").execution.query_method == "reference" &&
          cli_options({}, "optimized").execution.query_method == "optimized",
          "Conformer CLI ignored an explicit query environment");
    check(cli_options({"--query-method", "reference"}, "optimized").execution.query_method == "reference" &&
          cli_options({"--query-method", "optimized"}, "reference").execution.query_method == "optimized" &&
          cli_options({"--query-method", "reference"}, "invalid").execution.query_method == "reference",
          "An inherited query method overrode the explicit CLI");
    for (const auto* method : {"invalid", "", "REFERENCE"}) {
        bool rejected = false;
        try { (void)cli_options({}, method); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Conformer CLI accepted a noncanonical query environment");
    }
    const auto selected = cli_options({"--gpu-batch", "on", "--batch-size", "4", "--batch-memory-mib", "512",
        "--cpu-gpu-overlap", "on", "--rescoring-backend", "cuda", "--query-method", "optimized",
        "--ranking", "rmsd", "--fast-prefilter", "none", "--pose-center-distance", "on"});
    check(selected.execution.gpu_batch && selected.execution.batch_size == 4 &&
          selected.execution.batch_memory_bytes == std::size_t{512} * 1024 * 1024 &&
          selected.execution.cpu_gpu_overlap == true &&
          selected.execution.rescoring_backend == GpuRescoringBackend::Cuda &&
          selected.execution.query_method == "optimized" && selected.rmsd_oracle && !selected.prefilter &&
          selected.pose_center_distance,
          "Conformer CLI lost shared CUDA controls or changed the complete ranking pool");
    check(cli_options({"--query-method", "optimized", "--fast-prefilter", "500"}).prefilter == 500,
          "Explicit query policy rejected the legacy CPU prefilter protocol");
    check(!cli_options({"--pose-center-distance", "off"}).pose_center_distance,
          "Explicit off enabled the center diagnostic");
    for (const auto& flags : std::vector<std::vector<std::string>>{
            {"--gpu-batch", "on"}, {"--batch-size", "4"}, {"--batch-memory-mib", "512"},
            {"--gpu-batch", "on", "--rescoring-backend", "cuda", "--batch-size", "65"},
            {"--gpu-batch", "on", "--rescoring-backend", "cuda", "--batch-memory-mib", "0"},
            {"--gpu-batch", "on", "--rescoring-backend", "cuda", "--batch-memory-mib", "18446744073709551615"},
            {"--cpu-gpu-overlap", "invalid"}, {"--query-method", "invalid"}, {"--rescoring-backend", "invalid"},
            {"--ranking", "hybrid"}, {"--ranking", "rmsd", "--fast-prefilter", "500"},
            {"--rescoring-backend", "cuda", "--fast-prefilter", "500"},
            {"--cpu-gpu-overlap", "off", "--fast-prefilter", "500"},
            {"--pose-center-distance", "invalid"}}) {
        bool rejected = false;
        try { (void)cli_options(flags); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Conformer CLI accepted invalid controls or a filtered global protocol");
    }
}

void test_diagnostic_outputs(const fs::path& directory) {
    const auto options = diagnostic_cli({"--gpu-batch", "on", "--batch-size", "4"});
    check(options.diagnostics && options.generation.count == 20 && options.generation.seed == 20261027 &&
          options.generation.min_rmsd == 0.5 && !options.generation.optimize_mmff &&
          !options.execution.cpu_gpu_overlap.has_value() &&
          options.pose_export.mode == PoseExportMode::None && options.execution.batch_size == 4 &&
          options.execution.batch_memory_bytes == std::size_t{2048} * 1024 * 1024,
          "Diagnostics changed the fixed generation protocol or enabled overlap/export");
    for (const auto& flags : std::vector<std::vector<std::string>>{
             {"--mmff"}, {"--num-conformers", "5"}, {"--seed", "42"}, {"--min-rmsd", "0"},
             {"--generate-only"}, {"--original-only"}, {"--fast-prefilter", "500"},
             {"--ranking", "rmsd"}, {"--pose-output", "all"}, {"--pose-output", "topk"},
             {"--write-sdf"}, {"--query-method", "reference"}}) {
        bool rejected = false;
        try { (void)diagnostic_cli(flags); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Diagnostics accepted a filtered pool, pose export or a different generation protocol");
    }
    check(diagnostic_cli({"--pose-output", "none", "--num-conformers", "20"}).diagnostics,
          "Explicit diagnostics controls were rejected");
    const auto output = directory / "diagnostics";
    fs::create_directory(output);
    const std::vector<Record> records{synthetic_record(0), synthetic_record(7), synthetic_record(2)};
    const auto timings = write_diagnostic_outputs(output, "synthetic", records, records.front().molecule);
    check(timings.metrics_s >= 0 && timings.analysis_s >= 0, "Invalid diagnostic timing");
    const auto summary = table(output / "summary.csv"), topk = table(output / "topk_metrics.csv");
    check(summary.size() == 81 && topk.size() == 1601, "Diagnostics did not write all four pools and budgets");
    const auto conformers = table(output / "conformers.csv");
    check(conformers.size() == 4 && conformers.front().back() == "accepted_order" &&
          conformers[1].back().empty() && conformers[2].back() == "1" && conformers[3].back() == "2",
          "Accepted prefix order was confused with conformer ID order");
    DiagnosticProgress counts;
    counts.generated = counts.accepted = counts.valid = 2;
    counts.failed = 0; counts.with_survivors = 2;
    counts.original_survivors = 25; counts.total_survivors = 75;
    write_diagnostic_status(output, "ok", "complete", "", counts, 0.25);
    auto protocol = output_file(output / "protocol.csv");
    protocol << "key,value\n";
    write_gpu_docking_summary_entry(protocol, "diagnostics", "on");
    protocol.close();
    std::set<std::string> files;
    for (const auto& entry : fs::directory_iterator(output)) files.insert(entry.path().filename().string());
    check(files == std::set<std::string>{"summary.csv", "topk_metrics.csv", "conformers.csv", "protocol.csv", "status.csv"},
          "Diagnostics wrote pose/coordinate files or omitted an aggregate CSV");
    const auto failure = directory / "diagnostic-failure";
    fs::create_directory(failure);
    write_diagnostic_status(failure, "invalid_reference", "reference_validation", "invalid, \"reference\"", {}, 0.1);
    check(contents(failure / "status.csv").find("\"invalid, \"\"reference\"\"\"") != std::string::npos,
          "Diagnostic status failed to escape CSV messages");
}

void test_center_distance(const fs::path& directory) {
    const auto molecule = [](double x, double y, double z) {
        RDKit::RWMol result;
        result.addAtom(new RDKit::Atom(6), true, true);
        result.addAtom(new RDKit::Atom(8), true, true);
        result.addBond(0, 1, RDKit::Bond::SINGLE);
        auto conformer = std::make_unique<RDKit::Conformer>(2);
        conformer->setAtomPos(0, {x - 1.0, y, z});
        conformer->setAtomPos(1, {x + 1.0, y, z});
        result.addConformer(conformer.release(), true);
        return result;
    };
    const auto reference = molecule(0, 0, 0);
    std::vector<Record> records{synthetic_record(0), synthetic_record(1)};
    for (auto& record : records) {
        record.molecule = record.id ? molecule(0.1, 0.2, 0.3) : reference;
        record.docking.oracle_all_survivors.reset();
        for (auto& pose : record.docking.poses) {
            pose.scored.pose.transform.rotation = {0, -1, 0, 1, 0, 0, 0, 0, 1};
            GpuRigidTransform transform{};
            std::copy(pose.scored.pose.transform.rotation.begin(), pose.scored.pose.transform.rotation.end(), transform.rotation);
            std::copy(pose.scored.pose.transform.translation.begin(), pose.scored.pose.transform.translation.end(), transform.translation);
            transform.valid = 1;
            pose.rmsd = gpu_pose_heavy_atom_rmsd(record.molecule, transform, reference);
            if (!record.docking.oracle_all_survivors || pose.rmsd < *record.docking.oracle_all_survivors)
                record.docking.oracle_all_survivors = pose.rmsd;
        }
        record.docking.oracle_after_prefilter = record.docking.oracle_all_survivors;
    }
    const auto rankings = rank_populations(records, true);
    ConformerEnsemble ensemble;
    ensemble.requested = ensemble.embedded = ensemble.distinct = 1;
    for (const auto mode : {PoseExportMode::None, PoseExportMode::TopK, PoseExportMode::All}) {
        for (const auto detail : {PoseExportDetail::Compact, PoseExportDetail::Full}) {
            const auto name = std::string("center-") + pose_export_mode_name(mode) +
                (detail == PoseExportDetail::Compact ? "-compact" : "-full");
            const auto off = directory / (name + "-off"), on = directory / (name + "-on");
            fs::create_directory(off); fs::create_directory(on);
            Options options;
            options.pose_output_explicit = true;
            options.pose_export = {mode, 3, detail};
            write_rankings(off, records, rankings, options);
            write_summary(off / "summary.csv", rankings, ensemble, records, 1.0, 2.0);
            options.pose_center_distance = true;
            PoseCenterCache centers(reference);
            write_rankings(on, records, rankings, options, &centers);
            write_summary(on / "summary.csv", rankings, ensemble, records, 1.0, 2.0);
            check(contents(off / "summary.csv") == contents(on / "summary.csv"),
                  "Center export changed scores, RMSD, ranking or summary metrics");
            if (mode == PoseExportMode::None) {
                check(!fs::exists(off / "poses.csv") && !fs::exists(on / "poses.csv") &&
                      centers.distances.empty() && centers.ligand_centers.empty(),
                      "No-export mode computed a pose diagnostic or created a CSV");
                continue;
            }
            const auto before = table(off / "poses.csv"), after = table(on / "poses.csv");
            check(before.size() == after.size() && after.front().back() == "center_distance_A",
                  "Center diagnostic changed row count or omitted its column");
            std::set<std::pair<std::size_t, std::size_t>> identities;
            for (std::size_t i = 0; i < after.size(); ++i) {
                check(after[i].size() == before[i].size() + 1 &&
                      std::equal(before[i].begin(), before[i].end(), after[i].begin()),
                      "Center export changed an existing CSV field or transform");
                if (!i) continue;
                const auto conformer = std::stoul(after[i][2]), pose = std::stoul(after[i][3]);
                identities.emplace(conformer, pose);
                const auto translation = records.at(conformer).docking.poses.at(pose).scored.pose.transform.translation[0];
                const double expected = conformer ? std::sqrt((translation - 0.2) * (translation - 0.2) + 0.1) : translation;
                check(std::abs(std::stod(after[i].back()) - expected) < 1e-12,
                      "Exported center distance does not match the geometric centroid");
                check(std::stod(after[i].back()) < std::stod(after[i][12]),
                      "Center distance was replaced with atomwise RMSD");
            }
            check(centers.distances.size() == identities.size() && centers.ligand_centers.size() == 2,
                  "The four energy rankings and Oracle did not share a lazy pose/conformer cache");
            check(centers.distances.size() == (mode == PoseExportMode::All ? 50 : 7),
                  "Center cache evaluated poses outside the exported union");
        }
    }
    auto no_coordinates = reference;
    no_coordinates.clearConformers();
    bool rejected = false;
    try { PoseCenterCache invalid(no_coordinates); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "An enabled center diagnostic accepted a reference without coordinates");
}

void test_oracle_and_shared_timings(const fs::path& directory) {
    std::vector<Record> records{synthetic_record(0), synthetic_record(1)};
    records[0].docking.poses.front().rmsd = 0.0;
    records[0].docking.poses.front().scored.pose.transform.translation[0] = 0.0;
    records[0].docking.oracle_all_survivors = records[0].docking.oracle_after_prefilter = 0.0;
    records[1].docking.shared_search = records[1].docking.shared_rescoring = true;
    const auto rankings = rank_populations(records, true);
    check(rankings.size() == 9, "Oracle mode replaced an energy ranking or included the baseline");
    const auto oracle = std::find_if(rankings.begin(), rankings.end(), [](const auto& row) { return row.rmsd_oracle; });
    check(oracle != rankings.end() && oracle->population == "ensemble" && oracle->pool.size() == 25 &&
          oracle->ranked.size() == 25 && oracle->pool[oracle->ranked.front().source_index].conformer_id == 1 &&
          oracle->pool[oracle->ranked.front().source_index].pose_id == 24 &&
          oracle->ranked.front().score == 0.1,
          "RMSD oracle selected the crystal baseline or lost the best unfiltered pose");
    for (const auto& row : oracle->pool) check(row.conformer_id != 0, "Baseline entered the oracle pool");
    const auto output = directory / "oracle";
    fs::create_directory(output);
    Options options;
    options.rmsd_oracle = true;
    write_rankings(output, records, rankings, options);
    const auto poses = table(output / "poses.csv");
    const auto first = std::find_if(poses.begin() + 1, poses.end(), [](const auto& row) { return row[1] == "rmsd"; });
    check(first != poses.end() && (*first)[0] == "ensemble" && (*first)[2] == "1" && (*first)[3] == "24" &&
          std::stod((*first)[7]) == 0.1 && std::stod((*first)[12]) == 0.1,
          "Oracle export changed its method, identity or minimum-RMSD ordering");
    ConformerEnsemble ensemble;
    ensemble.requested = ensemble.embedded = ensemble.distinct = 1;
    write_summary(output / "summary.csv", rankings, ensemble, records, 1.0, 2.0);
    const auto summary = table(output / "summary.csv");
    check(summary.front().back() == "top1_minus_oracle_all_A", "Summary omitted the score-selection/oracle gap");
    for (std::size_t i = 1; i < summary.size(); ++i) {
        const auto& row = summary[i];
        check(row.size() == summary.front().size(), "Shared timings changed summary row width");
        if (row[0] == "ensemble")
            for (auto index : {21, 23, 24, 25})
                check(row[index].empty(), "A shared CUDA interval was presented as a measured population time");
        if (row[1] == "rmsd") check(std::stod(row.back()) == 0.0, "Oracle diagnostic gap must be zero");
    }
    write_conformers(output / "conformers.csv", records, false);
    const auto conformers = table(output / "conformers.csv");
    check(conformers.size() == 3 && conformers[2][16].empty() && conformers[2][18].empty() &&
          conformers[2][20].empty() && conformers[2][21].empty(),
          "Shared per-conformer CUDA intervals must be blank");
}

} // namespace

int main() {
    fs::path directory;
    try {
        auto pattern = (fs::temp_directory_path() / "conformer-export-test-XXXXXX").string();
        const char* created = ::mkdtemp(pattern.data());
        check(created != nullptr, "Cannot create isolated synthetic output directory");
        directory = created;
        test_outputs(directory);
        test_cli_protocols();
        test_diagnostic_outputs(directory);
        test_oracle_and_shared_timings(directory);
        test_center_distance(directory);
        fs::remove_all(directory);
        std::cout << "Conformer export tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (!directory.empty()) {
            std::error_code ignored;
            fs::remove_all(directory, ignored);
        }
        std::cerr << "Conformer export test failed: " << error.what() << '\n';
        return 1;
    }
}
