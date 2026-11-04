#include "pose_output.hpp"
#include "run_config.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Operation>
void rejects(Operation operation, const char* message) {
    try { operation(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

struct TemporaryDirectory {
    fs::path path = fs::temp_directory_path() /
        ("docking-pose-output-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

std::vector<std::string> read_lines(const fs::path& path) {
    std::ifstream input(path);
    require(input.good(), "Missing ranking CSV");
    std::vector<std::string> lines;
    for (std::string line; std::getline(input, line);) lines.push_back(line);
    return lines;
}

std::size_t columns(const std::string& row) {
    return 1 + std::count(row.begin(), row.end(), ',');
}

std::string read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "Missing output for byte comparison");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_input(const fs::path& path, const std::string& contents) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << contents;
}

std::string center_ligand(double x, double y, double hydrogen_x) {
    std::ostringstream output;
    output << "@<TRIPOS>MOLECULE\ncenter-test\n4 3 0 0 0\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n"
        << "1 C1 " << x << ' ' << y << " 0 C.3 1 LIG 0\n"
        << "2 C2 " << x + 2 << ' ' << y << " 0 C.3 1 LIG 0\n"
        << "3 O1 " << x + 1 << ' ' << y + 3 << " 0 O.3 1 LIG 0\n"
        << "4 H1 " << hydrogen_x << " 0 0 H 1 LIG 0\n"
        << "@<TRIPOS>BOND\n1 1 2 1\n2 2 3 1\n3 1 4 1\n";
    return output.str();
}

RunConfig parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return parse_run_config(static_cast<int>(argv.size()), argv.data());
}

void check_cpu_environment() {
    const auto* current = std::getenv("DOCKING_KABSCH_METHOD");
    const std::optional<std::string> saved = current ? std::optional<std::string>(current) : std::nullopt;
    const auto restore = [&] {
        if (saved) ::setenv("DOCKING_KABSCH_METHOD", saved->c_str(), 1);
        else ::unsetenv("DOCKING_KABSCH_METHOD");
    };
    try {
        require(::unsetenv("DOCKING_KABSCH_METHOD") == 0, "Cannot clear the test selector");
        apply_cpu_docking_environment();
        require(std::string(std::getenv("DOCKING_KABSCH_METHOD")) == "quaternion",
                "CPU docking did not select Quaternion by default");
        for (const auto* method : {"svd", "quaternion"}) {
            require(::setenv("DOCKING_KABSCH_METHOD", method, 1) == 0, "Cannot set the test selector");
            apply_cpu_docking_environment();
            require(std::string(std::getenv("DOCKING_KABSCH_METHOD")) == method,
                    "CPU docking overwrote an explicit Kabsch selector");
        }
    } catch (...) { restore(); throw; }
    restore();
}

void check_cli() {
    const auto defaults = parse({"test", "synthetic"});
    require(defaults.pose_export.mode == PoseExportMode::TopK &&
            defaults.pose_export.top_k == 20 && !defaults.pose_export.detail &&
            defaults.export_structures && !defaults.export_timings &&
            !defaults.pose_center_distance && defaults.center_reference_mol2.empty(),
            "Default export settings changed");
    const auto configured = parse({"test", "synthetic", "--pose-output", "all",
        "--top-k", "2", "--pose-detail", "full", "--no-structures", "--timings"});
    require(configured.pose_export.mode == PoseExportMode::All &&
            configured.pose_export.top_k == 2 &&
            configured.pose_export.detail == PoseExportDetail::Full &&
            !configured.export_structures && configured.export_timings, "Export options not parsed");
    const auto diagnostic = parse({"test", "synthetic", "--pose-center-distance", "on",
        "--center-reference-mol2", "reference.mol2"});
    require(diagnostic.pose_center_distance && diagnostic.center_reference_mol2 == "reference.mol2",
            "Center diagnostic options not parsed");
    const auto disabled = parse({"test", "--pose-center-distance", "off",
        "--center-reference-mol2", "missing.mol2"});
    require(!disabled.pose_center_distance, "Explicit OFF enabled center diagnostics");
    for (const auto& arguments : std::vector<std::vector<std::string>>{
             {"test", "--top-k", "0"}, {"test", "--top-k", "-1"},
             {"test", "--top-k", "2x"}, {"test", "--top-k"},
             {"test", "--pose-output", "invalid"},
             {"test", "--pose-detail", "invalid"},
             {"test", "--pose-center-distance", "on"},
             {"test", "--pose-center-distance", "invalid"},
             {"test", "--pose-center-distance"},
             {"test", "--center-reference-mol2"},
             {"test", "--center-reference-mol2", ""},
             {"test", "--center-reference-mol2", "--quiet"}}) {
        bool rejected = false;
        try { parse(arguments); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Invalid export option accepted");
    }
}

void check_center_distance() {
    TemporaryDirectory temporary;
    const auto ligand = temporary.path / "ligand.mol2";
    const auto reference = temporary.path / "reference.mol2";
    write_input(ligand, center_ligand(0, 0, 50));
    write_input(reference, center_ligand(3, 4, 150));
    RunResult result;
    result.complex_id = "center-test";
    for (std::size_t i = 0; i < 4; ++i) {
        RankedPoseDiagnostic pose;
        pose.rmsd = 9.0 - i;
        pose.fast_weighted_score = 20.0 - i;
        pose.weighted_interaction_score.weighted_score = 10.0 - i;
        pose.transform.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (i == 1) pose.transform.translation = {3, 4, 0};
        if (i == 2) pose.transform.translation = {3, 0, 0};
        if (i == 3) {
            pose.transform.rotation = {0, -1, 0, 1, 0, 0, 0, 0, 1};
            pose.transform.translation = {5, 4, 0};
        }
        result.pose_ranking.weighted_interaction_ranking.push_back(pose);
    }
    RunConfig config;
    config.export_structures = false;
    std::size_t run = 0;
    for (const auto detail : {std::optional<PoseExportDetail>{},
                              std::optional{PoseExportDetail::Compact},
                              std::optional{PoseExportDetail::Full}}) {
        for (const auto mode : {PoseExportMode::TopK, PoseExportMode::All}) {
            config.pose_export = {mode, 2, detail};
            config.pose_center_distance = false;
            config.ligand_path = temporary.path / "missing-ligand.mol2";
            config.center_reference_mol2.clear();
            config.output_directory = temporary.path / ("off-" + std::to_string(++run));
            save_pose_outputs(result, config, 2);
            const auto original = config.output_directory / result.complex_id / "ranking.csv";
            const auto unchanged = read_bytes(original);
            const auto off_rows = read_lines(original);

            config.center_reference_mol2 = temporary.path / "missing-reference.mol2";
            config.output_directory = temporary.path / ("explicit-off-" + std::to_string(run));
            save_pose_outputs(result, config, 2);
            require(read_bytes(config.output_directory / result.complex_id / "ranking.csv") == unchanged,
                    "Disabled center diagnostics read inputs or changed CSV bytes");

            config.pose_center_distance = true;
            config.ligand_path = ligand;
            config.center_reference_mol2 = reference;
            config.output_directory = temporary.path / ("on-" + std::to_string(run));
            save_pose_outputs(result, config, 2);
            const auto rows = read_lines(config.output_directory / result.complex_id / "ranking.csv");
            require(rows.size() == off_rows.size() && rows.front() == off_rows.front() + ",center_distance_A",
                    "Center diagnostic changed the CSV schema beyond its appended field");
            const std::array<double, 4> expected{5, 0, 4, 0};
            for (std::size_t i = 1; i < rows.size(); ++i) {
                const auto last = rows[i].rfind(',');
                require(rows[i].substr(0, last) == off_rows[i] &&
                        columns(rows[i]) == columns(off_rows[i]) + 1,
                        "Center diagnostics changed existing values, transforms or pose order");
                const auto distance = std::stod(rows[i].substr(last + 1));
                require(std::isfinite(distance) && std::abs(distance - expected[i - 1]) <= 1e-12,
                        "Center distance did not use the transformed heavy-atom centroid");
            }
        }
    }

    config.pose_export.mode = PoseExportMode::None;
    config.output_directory = temporary.path / "none";
    save_pose_outputs(result, config);
    require(!fs::exists(config.output_directory), "None mode created diagnostic output files");
    config.center_reference_mol2.clear();
    rejects([&] { save_pose_outputs(result, config); }, "Missing explicit reference accepted in None mode");
    config.center_reference_mol2 = temporary.path / "missing.mol2";
    rejects([&] { save_pose_outputs(result, config); }, "Missing reference file accepted in None mode");
    config.center_reference_mol2 = temporary.path / "invalid.mol2";
    write_input(config.center_reference_mol2, "not a MOL2 molecule\n");
    rejects([&] { save_pose_outputs(result, config); }, "Malformed reference accepted in None mode");
    config.center_reference_mol2 = temporary.path / "multiple.mol2";
    write_input(config.center_reference_mol2, center_ligand(3, 4, 150) + center_ligand(0, 0, 50));
    rejects([&] { save_pose_outputs(result, config); }, "Multiple reference records accepted in None mode");
    config.center_reference_mol2 = reference;
    config.ligand_path = temporary.path / "missing-ligand.mol2";
    rejects([&] { save_pose_outputs(result, config); }, "Missing ligand accepted by an enabled diagnostic");
    config.ligand_path = ligand;
    config.center_reference_mol2 = temporary.path / "hydrogen-only.mol2";
    write_input(config.center_reference_mol2,
        "@<TRIPOS>MOLECULE\nhydrogen-only\n1 0 0 0 0\nSMALL\nUSER_CHARGES\n\n"
        "@<TRIPOS>ATOM\n1 H1 0 0 0 H 1 LIG 0\n@<TRIPOS>BOND\n");
    rejects([&] { save_pose_outputs(result, config); }, "Reference without heavy atoms accepted");
    config.pose_center_distance = false;
    save_pose_outputs(result, config);
    require(!fs::exists(config.output_directory), "OFF/None accessed or created output");

    config.pose_center_distance = true;
    config.center_reference_mol2 = reference;
    config.pose_export.mode = PoseExportMode::All;
    config.output_directory = temporary.path / "empty-centers";
    result.pose_ranking.weighted_interaction_ranking.clear();
    save_pose_outputs(result, config);
    const auto empty = read_lines(config.output_directory / result.complex_id / "ranking.csv");
    require(empty.size() == 1 && empty.front().ends_with(",center_distance_A"),
            "Empty ranking lost the optional center-distance schema");
}

void check_exports() {
    TemporaryDirectory temporary;
    RunResult result;
    result.complex_id = "synthetic";
    result.pose_generation.total_candidate_matches = 17;
    result.pose_generation.transforms_generated = 19;
    result.pose_generation.best_rmsd_below_clash_threshold[2] = 0.125;
    result.timings.online_total_seconds = 1.25;
    for (std::size_t i = 0; i < 4; ++i) {
        RankedPoseDiagnostic pose;
        pose.rmsd = 3.0 - i;
        pose.fast_weighted_score = 20.0 - i;
        pose.weighted_interaction_score.weighted_score = 10.0 - i;
        pose.weighted_interaction_score.total_interactions = 7;
        pose.weighted_interaction_score.satisfied_ligand_areas = 3;
        pose.compatible_areas = 5;
        pose.occupied_atoms = 1;
        pose.clash_fraction = 0.025;
        pose.transform.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        pose.transform.translation = {static_cast<double>(i) + 0.125, -0.25, 0.5};
        result.pose_ranking.weighted_interaction_ranking.push_back(pose);
    }
    const auto before = result;
    RunConfig config;
    // Missing inputs make accidental molecule loading or reference copying fail.
    config.protein_path = temporary.path / "missing_protein.pdb";
    config.pocket_path = temporary.path / "missing_pocket.pdb";
    config.ligand_path = temporary.path / "missing_ligand.mol2";
    config.output_directory = temporary.path / "none";
    config.pose_export.mode = PoseExportMode::None;
    save_pose_outputs(result, config);
    require(!fs::exists(temporary.path), "None mode accessed or created output");

    config.export_structures = false;
    config.pose_export.mode = PoseExportMode::TopK;
    config.output_directory = temporary.path / "legacy";
    save_pose_outputs(result, config, 2);
    auto rows = read_lines(config.output_directory / result.complex_id / "ranking.csv");
    require(rows.size() == 3 && rows.front() ==
        "rank,rmsd,weighted_interaction_score,interaction_count,satisfied_ligand_areas,"
        "fast_weighted_score,grid_compatible_areas,occupied_atoms,clash_fraction",
        "Legacy CSV schema or top_n changed");
    require(rows[1] == "1,3,10,7,3,20,5,1,0.025", "Legacy CSV values or order changed");

    config.pose_export.detail = PoseExportDetail::Compact;
    config.output_directory = temporary.path / "compact";
    save_pose_outputs(result, config, 1);
    rows = read_lines(config.output_directory / result.complex_id / "ranking.csv");
    require(rows.size() == 2 && columns(rows.front()) == 6 && columns(rows[1]) == 6,
            "Compact CSV must contain only the essential six fields");
    require(rows[1].starts_with("1,3,10,20,1,"), "Compact export changed selected pose");

    config.pose_export.mode = PoseExportMode::All;
    config.pose_export.detail = PoseExportDetail::Full;
    config.output_directory = temporary.path / "full";
    save_pose_outputs(result, config, 1);
    const auto output = config.output_directory / result.complex_id;
    rows = read_lines(output / "ranking.csv");
    require(rows.size() == 5 && columns(rows.front()) == 21,
            "All mode must write every available ranked pose, regardless of top_n");
    require(rows.front().ends_with("r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz") &&
            rows[1].ends_with(",1,0,0,0,1,0,0,0,1,0.125,-0.25,0.5") &&
            rows[4].ends_with(",1,0,0,0,1,0,0,0,1,3.125,-0.25,0.5"),
            "Full export lost transforms or changed pose order");
    require(std::distance(fs::directory_iterator(output), fs::directory_iterator{}) == 1,
            "No-structures mode wrote molecular or visualization files");

    require(result.pose_generation.total_candidate_matches == before.pose_generation.total_candidate_matches &&
            result.pose_generation.transforms_generated == before.pose_generation.transforms_generated &&
            result.pose_generation.best_rmsd_below_clash_threshold == before.pose_generation.best_rmsd_below_clash_threshold &&
            result.timings.online_total_seconds == before.timings.online_total_seconds,
            "Export changed scientific metrics");
    for (std::size_t i = 0; i < before.pose_ranking.weighted_interaction_ranking.size(); ++i) {
        const auto& old_pose = before.pose_ranking.weighted_interaction_ranking[i];
        const auto& pose = result.pose_ranking.weighted_interaction_ranking[i];
        require(pose.transform.rotation == old_pose.transform.rotation &&
                pose.transform.translation == old_pose.transform.translation &&
                pose.rmsd == old_pose.rmsd && pose.fast_weighted_score == old_pose.fast_weighted_score &&
                pose.weighted_interaction_score.weighted_score == old_pose.weighted_interaction_score.weighted_score,
                "Export changed scored pose data");
    }
    result.pose_ranking.weighted_interaction_ranking.clear();
    config.output_directory = temporary.path / "empty";
    save_pose_outputs(result, config);
    require(read_lines(config.output_directory / result.complex_id / "ranking.csv").size() == 1,
            "Empty ranking must still write its CSV header");
}

} // namespace

int main() {
    try {
        check_cpu_environment();
        check_cli();
        check_exports();
        check_center_distance();
        std::cout << "Pose output tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Pose output test: " << error.what() << '\n';
        return 1;
    }
}
