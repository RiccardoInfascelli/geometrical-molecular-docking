#include "conformer_generation.hpp"
#include "conformer_quality_input.hpp"
#include "quaternion_rmsd.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <RDGeneral/versions.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace conformer_experiment;

namespace {
constexpr std::array<std::size_t, 7> budgets{1, 5, 10, 20, 50, 100, 200};
struct Options {
    fs::path ligand, output;
    std::size_t count = 200;
    int seed = 20261027;
    bool allow_undefined_stereo = false;
    bool write_coordinates = true;
};

unsigned int integer(const std::string& value) {
    unsigned int result{};
    const auto converted = std::from_chars(value.data(), value.data() + value.size(), result);
    if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size())
        throw std::invalid_argument("Expected a nonnegative integer: " + value);
    return result;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (name == "--allow-undefined-stereo") { options.allow_undefined_stereo = true; continue; }
        if (name == "--no-coordinates") { options.write_coordinates = false; continue; }
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + name);
        const std::string value = argv[++i];
        if (name == "--ligand") options.ligand = value;
        else if (name == "--output") options.output = value;
        else if (name == "--max-conformers") options.count = integer(value);
        else if (name == "--seed") {
            const auto seed = integer(value);
            if (seed > static_cast<unsigned int>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Seed exceeds INT_MAX");
            options.seed = static_cast<int>(seed);
        } else throw std::invalid_argument("Unknown option: " + name);
    }
    if (options.ligand.empty() || options.output.empty() || !options.count || options.count > 200)
        throw std::invalid_argument("Require --ligand, --output and max-conformers in [1,200]");
    return options;
}

std::string csv(const std::string& value) {
    std::string escaped = "\"";
    for (auto c : value) { if (c == '"') escaped += '"'; escaped += c; }
    return escaped + '"';
}

std::string json(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}

std::ofstream output(const fs::path& path) {
    std::ofstream file(path);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << std::setprecision(17);
    return file;
}

double seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

void value(std::ostream& out, const std::optional<double>& item) { if (item) out << *item; }
void value(std::ostream& out, const std::optional<std::size_t>& item) { if (item) out << *item; }

int run(const Options& options) {
    const auto begin = Clock::now();
    if (!fs::create_directories(options.output))
        throw std::invalid_argument("Output directory already exists; use a new directory");
    auto conformers = output(options.output / "conformers.csv");
    auto mappings = output(options.output / "atom_mapping.csv");
    std::ofstream coordinates;
    if (options.write_coordinates) coordinates = output(options.output / "conformer_coordinates.csv");
    auto budget_file = output(options.output / "budgets.csv");
    auto summary = output(options.output / "summary.csv");
    conformers << "sequence_index,rdkit_conformer_id,aligned_heavy_rmsd,success_lt_1,success_lt_1_5,success_lt_2\n";
    mappings << "atom_index,mol2_atom_id,atom_name,atomic_number,heavy_index\n";
    if (options.write_coordinates) coordinates << "sequence_index,atom_index,x,y,z\n";
    budget_file << "budget,available,complete,best_rmsd,success_lt_1,success_lt_1_5,success_lt_2,status\n";
    summary << "status,success_lt_2,requested,generated,heavy_atoms,best_rmsd,first_lt_1,first_lt_1_5,first_lt_2,"
               "undefined_stereo,native_defined_stereo,seed,prepare_s,generation_s,evaluation_s,wall_s,error\n";

    std::string status = "error", error, graph_smiles;
    std::vector<double> rmsds;
    std::size_t heavy_atoms = 0, undefined_stereo = 0, native_defined_stereo = 0;
    int generation_calls = 0;
    double prepare_s = 0, generation_s = 0, evaluation_s = 0;
    auto stage = Clock::now();
    try {
        std::ifstream input(options.ligand);
        if (!input) throw std::runtime_error("Cannot read ligand MOL2");
        std::ostringstream block;
        block << input.rdbuf();
        if (input.bad()) throw std::runtime_error("Error reading ligand MOL2");
        const auto prepared = prepare_quality_input(block.str());
        heavy_atoms = prepared.heavy_indices.size();
        undefined_stereo = prepared.undefined_stereo;
        native_defined_stereo = prepared.native_defined_stereo;
        graph_smiles = RDKit::MolToSmiles(prepared.graph);
        std::size_t heavy_index = 0;
        for (unsigned int i = 0; i < prepared.graph.getNumAtoms(); ++i) {
            const auto* atom = prepared.graph.getAtomWithIdx(i);
            mappings << i << ',' << prepared.atom_ids[i] << ',' << csv(atom->getProp<std::string>("_TriposAtomName"))
                     << ',' << atom->getAtomicNum() << ',';
            if (atom->getAtomicNum() > 1) mappings << heavy_index++;
            mappings << '\n';
        }
        prepare_s = seconds(stage);
        if (undefined_stereo && !options.allow_undefined_stereo) {
            status = "stereo_ambiguous";
            error = "Graph has unspecified potential stereo; native coordinates are not used to resolve it";
        } else {
            ConformerOptions generation;
            generation.count = options.count;
            generation.seed = options.seed;
            generation.min_rmsd = 0;
            generation.compute_diversity = false;
            generation.optimize_mmff = false;
            stage = Clock::now();
            ConformerEnsemble ensemble;
            ++generation_calls;
            try { ensemble = generate_conformers(prepared.graph, generation); }
            catch (...) { generation_s = seconds(stage); status = "generation_failed"; throw; }
            generation_s = seconds(stage);
            stage = Clock::now();
            const auto reference = heavy_coordinates(prepared.reference, prepared.heavy_indices);
            for (const auto& conformer : ensemble.conformers) {
                const auto rmsd = quaternion_aligned_rmsd(
                    heavy_coordinates(conformer.molecule, prepared.heavy_indices), reference);
                rmsds.push_back(rmsd);
                conformers << rmsds.size() << ',' << conformer.id << ',' << rmsd << ','
                           << (rmsd < 1) << ',' << (rmsd < 1.5) << ',' << (rmsd < 2) << '\n';
                if (options.write_coordinates) {
                    const auto& conf = conformer.molecule.getConformer();
                    for (unsigned int i = 0; i < conformer.molecule.getNumAtoms(); ++i) {
                        const auto& p = conf.getAtomPos(i);
                        coordinates << rmsds.size() << ',' << i << ',' << p.x << ',' << p.y << ',' << p.z << '\n';
                    }
                }
            }
            evaluation_s = seconds(stage);
            status = rmsds.size() == options.count ? "ok" : "incomplete";
        }
    } catch (const std::exception& e) {
        error = e.what();
        if (prepare_s == 0) prepare_s = seconds(stage);
    }

    for (auto budget : budgets) {
        const auto available = std::min(budget, rmsds.size());
        const bool complete = available == budget;
        budget_file << budget << ',' << available << ',' << complete << ',';
        if (available) {
            const auto best = *std::min_element(rmsds.begin(), rmsds.begin() + available);
            budget_file << best << ',' << (best < 1) << ',' << (best < 1.5) << ',' << (best < 2);
        } else budget_file << ",,,";
        const std::string budget_status = status == "ok" || status == "incomplete"
            ? (complete ? "complete" : (budget > options.count ? "not_requested" : "incomplete")) : status;
        budget_file << ',' << budget_status << '\n';
    }
    std::optional<double> best;
    std::array<std::optional<std::size_t>, 3> first;
    for (std::size_t i = 0; i < rmsds.size(); ++i) {
        best = best ? std::min(*best, rmsds[i]) : rmsds[i];
        for (std::size_t t = 0; t < first.size(); ++t)
            if (!first[t] && rmsds[i] < std::array<double, 3>{1, 1.5, 2}[t]) first[t] = i + 1;
    }
    summary << status << ',';
    if (best) summary << (*best < 2);
    summary << ',' << options.count << ',' << rmsds.size() << ',' << heavy_atoms << ',';
    value(summary, best);
    for (const auto& item : first) { summary << ','; value(summary, item); }
    summary << ',' << undefined_stereo << ',' << native_defined_stereo << ',' << options.seed << ','
            << prepare_s << ',' << generation_s << ',' << evaluation_s << ',' << seconds(begin) << ',' << csv(error) << '\n';

    // A readable terminal protocol must never precede the completed data files:
    // the runner uses it to distinguish finished cases from interrupted writes.
    conformers.close();
    mappings.close();
    if (options.write_coordinates) coordinates.close();
    budget_file.close();
    summary.close();
    auto protocol = output(options.output / "protocol.json");
    const auto& defaults = RDKit::DGeomHelpers::ETKDGv3;
    protocol << "{\n  \"protocol_version\": 1,\n  \"rdkit_version\": " << json(RDKit::rdkitVersion)
             << ",\n  \"compiler\": " << json(__VERSION__)
             << ",\n  \"ligand\": " << json(fs::absolute(options.ligand).string())
             << ",\n  \"generator\": \"ETKDGv3\",\n  \"seed\": " << options.seed
             << ",\n  \"max_conformers\": " << options.count
             << ",\n  \"coordinates_export\": " << (options.write_coordinates ? "true" : "false")
             << ",\n  \"coordinate_export_bytes\": " << (options.write_coordinates ?
                    fs::file_size(options.output / "conformer_coordinates.csv") : 0)
             << ",\n  \"budgets\": [1,5,10,20,50,100,200],\n  \"num_threads\": 1,\n"
                "  \"generation_calls\": " << generation_calls << ",\n  \"prefix_order\": \"RDKit returned conformer order\",\n"
                "  \"prune_rms_thresh\": -1,\n  \"diversity_filter\": false,\n  \"mmff\": false,\n"
                "  \"native_coordinates_in_generation\": false,\n  \"native_stereo_in_generation\": false,\n"
                "  \"generation_graph_smiles\": " << json(graph_smiles)
             << ",\n  \"allow_undefined_stereo\": " << (options.allow_undefined_stereo ? "true" : "false")
             << ",\n  \"undefined_stereo\": " << undefined_stereo
             << ",\n  \"alignment\": \"Horn quaternion, double eigensolve, wide accumulation, proper rotation\",\n"
                "  \"atom_mapping\": \"fixed original heavy-atom indices, no symmetry permutations\",\n"
                "  \"threshold_comparison\": \"strictly less than, angstrom\",\n"
                "  \"ETversion\": " << defaults.ETversion
             << ",\n  \"useExpTorsionAnglePrefs\": " << int(defaults.useExpTorsionAnglePrefs)
             << ",\n  \"useBasicKnowledge\": " << int(defaults.useBasicKnowledge)
             << ",\n  \"useSmallRingTorsions\": " << int(defaults.useSmallRingTorsions)
             << ",\n  \"useMacrocycleTorsions\": " << int(defaults.useMacrocycleTorsions)
             << ",\n  \"useMacrocycle14config\": " << int(defaults.useMacrocycle14config)
             << ",\n  \"useRandomCoords\": " << int(defaults.useRandomCoords)
             << ",\n  \"enforceChirality\": true,\n  \"coordMap\": null,\n"
                "  \"embedding_maxIterations\": " << defaults.maxIterations
             << ",\n  \"status\": " << json(status) << "\n}\n";
    protocol.close();
    std::cout << status << ": " << rmsds.size() << '/' << options.count << " conformers\n";
    if (!error.empty()) std::cerr << error << '\n';
    return status == "error" || status == "generation_failed" ? 1 : 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: conformer_quality --ligand single.mol2 --output NEW_DIRECTORY\n"
                     "       [--max-conformers 200] [--seed 20261027] [--allow-undefined-stereo]\n"
                     "       [--no-coordinates]  Keep all RMSD metrics; omit coordinate CSV only.\n";
        return 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
