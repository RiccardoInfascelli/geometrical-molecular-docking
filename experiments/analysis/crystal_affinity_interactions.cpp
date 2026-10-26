// Crystal contacts versus experimental PDBbind binding affinity.
// Analyzes original crystallographic coordinates; does not perform docking.
#include "interaction_areas.hpp"
#include "interaction_scoring.hpp"
#include "molecule_io.hpp"

#include <interaction/interaction_area.hpp>

#include <GraphMol/Atom.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Record {
    std::string pdb_id;
    double p_affinity{};
    std::string assay_type;
    std::string assay_raw;
};

// The columns are intentionally fixed, including error rows (blank numeric fields).
const std::vector<std::string> kColumns = {
    "pdb_id", "p_affinity", "assay_type", "assay_raw",
    "ligand_heavy_atoms", "protein_atoms",
    "ligand_areas", "protein_areas",
    "satisfied_ligand_areas", "satisfied_protein_areas", "satisfied_fraction",
    "hydrophobic", "hydrogen_bonds", "halogen_bonds", "ionic",
    "cation_pi", "pi_stacking", "metal", "total_interactions",
    "ligand_hydrophobic_areas", "ligand_donor_areas", "ligand_acceptor_areas",
    "ligand_halogen_donor_areas", "ligand_halogen_acceptor_areas",
    "ligand_cation_areas", "ligand_anion_areas", "ligand_aromatic_areas",
    "ligand_metal_areas", "ligand_chelated_areas",
    "status", "error"
};

std::string quote_csv(const std::string& value) {
    std::string result = "\"";
    for (char c : value) {
        if (c == '"') result += "\"\"";
        else result += c;
    }
    return result + '"';
}

void write_csv(std::ofstream& out, const std::vector<std::string>& row) {
    if (row.size() != kColumns.size())
        throw std::runtime_error("Internal CSV column mismatch");
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (i) out << ',';
        out << quote_csv(row[i]);
    }
    out << '\n';
}

template<typename T> std::string value_string(T value) {
    return std::to_string(value);
}

std::string decimal_string(double value) {
    std::ostringstream s;
    s << std::setprecision(12) << value;
    return s.str();
}

// PDBbind refined index format:
// 2r58  2.00  2007  2.00  Kd=10mM  // ...
std::vector<Record> load_index(const fs::path& index_path) {
    std::ifstream in(index_path);
    if (!in) throw std::runtime_error("Cannot open index: " + index_path.string());
    std::vector<Record> records;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Record item;
        std::string resolution, year, p_text;
        if (!(ss >> item.pdb_id >> resolution >> year >> p_text >> item.assay_raw))
            continue;
        try {
            std::size_t consumed = 0;
            item.p_affinity = std::stod(p_text, &consumed);
            if (consumed != p_text.size() || !std::isfinite(item.p_affinity))
                continue;
        } catch (const std::exception&) {
            continue;
        }
        // Exclude censored/approximate measurements, which are not point values.
        if (item.assay_raw.find_first_of("<>~") != std::string::npos)
            continue;
        if (item.assay_raw.rfind("Kd=", 0) == 0)
            item.assay_type = "Kd";
        else if (item.assay_raw.rfind("Ki=", 0) == 0)
            item.assay_type = "Ki";
        else
            continue;
        records.push_back(std::move(item));
    }
    if (records.empty()) throw std::runtime_error("No valid Kd/Ki records in index");
    return records;
}

struct LigandFeatures {
    std::size_t hydrophobic{}, donor{}, acceptor{}, halogen_donor{},
                halogen_acceptor{}, cation{}, anion{}, aromatic{}, metal{}, chelated{};
};

LigandFeatures count_ligand_features(
    const std::vector<interaction::interaction_area>& areas) {
    using P = interaction::interaction_smarts_pattern;
    using C = interaction::interaction_class_pattern;
    LigandFeatures out;
    for (const auto& area : areas) {
        if (area.pattern_class == C::Aromatic) {
            ++out.aromatic;
            continue;
        }
        switch (area.pattern) {
            case P::Hydrophobic: ++out.hydrophobic; break;
            case P::Hydrogen_donor: ++out.donor; break;
            case P::Hydrogen_acceptor: ++out.acceptor; break;
            case P::Halogen_donor: ++out.halogen_donor; break;
            case P::Halogen_acceptor: ++out.halogen_acceptor; break;
            case P::Cation: ++out.cation; break;
            case P::Anion: ++out.anion; break;
            case P::Metal: ++out.metal; break;
            case P::Chelated: ++out.chelated; break;
            default: break;
        }
    }
    return out;
}

std::size_t count_heavy_atoms(const RDKit::ROMol& molecule) {
    std::size_t n = 0;
    for (unsigned int i = 0; i < molecule.getNumAtoms(); ++i)
        if (molecule.getAtomWithIdx(i)->getAtomicNum() > 1) ++n;
    return n;
}

std::vector<std::string> analyze(const fs::path& root, const Record& record) {
    std::vector<std::string> row(kColumns.size());
    row[0] = record.pdb_id;
    row[1] = decimal_string(record.p_affinity);
    row[2] = record.assay_type;
    row[3] = record.assay_raw;
    const fs::path dir = root / record.pdb_id;
    const fs::path protein_path = dir / (record.pdb_id + "_protein.pdb");
    const fs::path ligand_path = dir / (record.pdb_id + "_ligand.mol2");

    try {
        if (!fs::is_regular_file(protein_path))
            throw std::runtime_error("Missing protein: " + protein_path.string());
        if (!fs::is_regular_file(ligand_path))
            throw std::runtime_error("Missing ligand: " + ligand_path.string());

        auto protein = load_protein(protein_path.string());
        auto ligand = load_ligand(ligand_path.string());
        if (!protein || !ligand) throw std::runtime_error("Molecule load returned null");

        const auto protein_areas = get_protein_areas(*protein);
        const auto ligand_areas = get_ligand_areas(*ligand);
        const auto weighted_interaction = score_weighted_interactions(protein_areas, ligand_areas);
        const auto features = count_ligand_features(ligand_areas);

        row[4] = value_string(count_heavy_atoms(*ligand));
        row[5] = value_string(protein->getNumAtoms());
        row[6] = value_string(ligand_areas.size());
        row[7] = value_string(protein_areas.size());
        row[8] = value_string(weighted_interaction.satisfied_ligand_areas);
        row[9] = value_string(weighted_interaction.satisfied_protein_areas);
        if (!ligand_areas.empty())
            row[10] = decimal_string(
                static_cast<double>(weighted_interaction.satisfied_ligand_areas) /
                static_cast<double>(ligand_areas.size()));
        row[11] = value_string(weighted_interaction.hydrophobic);
        row[12] = value_string(weighted_interaction.hydrogen_bonds);
        row[13] = value_string(weighted_interaction.halogen_bonds);
        row[14] = value_string(weighted_interaction.ionic);
        row[15] = value_string(weighted_interaction.cation_pi);
        row[16] = value_string(weighted_interaction.pi_stacking);
        row[17] = value_string(weighted_interaction.metal);
        row[18] = value_string(weighted_interaction.total_interactions);
        row[19] = value_string(features.hydrophobic);
        row[20] = value_string(features.donor);
        row[21] = value_string(features.acceptor);
        row[22] = value_string(features.halogen_donor);
        row[23] = value_string(features.halogen_acceptor);
        row[24] = value_string(features.cation);
        row[25] = value_string(features.anion);
        row[26] = value_string(features.aromatic);
        row[27] = value_string(features.metal);
        row[28] = value_string(features.chelated);
        row[29] = "ok";
    } catch (const std::exception& e) {
        row[29] = "error";
        row[30] = e.what();
    }
    return row;
}

void usage() {
    std::cerr << "Usage: crystal_affinity_interactions <refined-set-dir> "
              << "<INDEX_refined_data.2020> <output.csv> "
              << "[--limit N] [--only 1a1e]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) { usage(); return 2; }
    try {
        const fs::path root = argv[1], index_path = argv[2], output = argv[3];
        if (!fs::is_directory(root))
            throw std::runtime_error("Not a refined-set directory: " + root.string());
        std::size_t limit = std::numeric_limits<std::size_t>::max();
        std::string only;
        for (int i = 4; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--limit" && i + 1 < argc) {
                limit = std::stoull(argv[++i]);
                if (limit == 0) throw std::runtime_error("--limit must be positive");
            } else if (flag == "--only" && i + 1 < argc) {
                only = argv[++i];
            } else {
                usage();
                throw std::runtime_error("Unknown or incomplete option: " + flag);
            }
        }
        std::ofstream csv(output);
        if (!csv) throw std::runtime_error("Cannot create CSV: " + output.string());
        write_csv(csv, kColumns);
        auto records = load_index(index_path);
        std::size_t seen = 0, success = 0, failed = 0;
        bool found_only = false;
        for (const auto& record : records) {
            if (!only.empty() && record.pdb_id != only) continue;
            found_only = true;
            const auto row = analyze(root, record);
            write_csv(csv, row);
            if (row[29] == "ok") ++success;
            else {
                ++failed;
                std::cerr << record.pdb_id << ": " << row[30] << '\n';
            }
            ++seen;
            if (seen >= limit) break;
            if (seen % 100 == 0) std::cerr << "Processed " << seen << " complexes\n";
        }
        csv.close();
        if (!found_only)
            throw std::runtime_error("Requested PDB ID not in index: " + only);
        std::cout << "Analyzed=" << seen << " OK=" << success
                  << " errors=" << failed << " CSV=" << output << '\n';
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
