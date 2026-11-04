#include "conformer_quality_input.hpp"
#include "molecule_io.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Chirality.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/MolOps.h>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

namespace conformer_experiment {

QualityInput prepare_quality_input(const std::string& mol2) {
    QualityInput result;
    std::istringstream input(mol2);
    std::ostringstream neutral;
    std::string line;
    bool atoms = false;
    unsigned int records = 0;
    std::vector<std::string> atom_names;
    std::set<unsigned int> ids;
    while (std::getline(input, line)) {
        if (line.rfind("@<TRIPOS>MOLECULE", 0) == 0) ++records;
        if (line.rfind("@<TRIPOS>", 0) == 0)
            atoms = line.rfind("@<TRIPOS>ATOM", 0) == 0;
        else if (atoms && line.find_first_not_of(" \t\r") != std::string::npos && line[0] != '#') {
            std::istringstream row(line);
            unsigned int id;
            std::string name, x, y, z, type, remaining;
            if (!(row >> id >> name >> x >> y >> z >> type) || !id || !ids.insert(id).second)
                throw std::invalid_argument("Malformed or duplicate MOL2 atom identifier");
            std::getline(row, remaining);
            neutral << id << ' ' << name << " 0 0 0 " << type << remaining << '\n';
            result.atom_ids.push_back(id);
            atom_names.push_back(name);
            continue;
        }
        neutral << line << '\n';
    }
    if (records != 1 || atom_names.empty())
        throw std::invalid_argument("Expected exactly one nonempty MOL2 record");

    auto graph = parse_ligand_mol2_block(neutral.str());
    graph->clearConformers();
    // MOL2 commonly encodes chirality through 3D coordinates. Never recover
    // those labels from the crystal, including parser-assigned stereo flags.
    RDKit::MolOps::removeStereochemistry(*graph);
    for (auto* bond : graph->bonds()) bond->setBondDir(RDKit::Bond::NONE);
    result.graph = *graph;
    auto probe = result.graph;
    for (const auto& info : RDKit::Chirality::findPotentialStereo(probe, false, true))
        if (info.specified != RDKit::Chirality::StereoSpecified::Specified)
            ++result.undefined_stereo;

    auto reference = parse_ligand_mol2_block(mol2);
    result.reference = *reference;
    if (graph->getNumAtoms() != reference->getNumAtoms() || graph->getNumAtoms() != atom_names.size() ||
        graph->getNumBonds() != reference->getNumBonds())
        throw std::runtime_error("Coordinate neutralization changed atom/bond counts");
    for (unsigned int i = 0; i < graph->getNumAtoms(); ++i) {
        const auto* a = graph->getAtomWithIdx(i);
        const auto* b = reference->getAtomWithIdx(i);
        if (a->getAtomicNum() != b->getAtomicNum() || a->getIsotope() != b->getIsotope() ||
            a->getFormalCharge() != b->getFormalCharge() || a->getIsAromatic() != b->getIsAromatic() ||
            a->getAtomMapNum() != b->getAtomMapNum() ||
            a->getNumExplicitHs() != b->getNumExplicitHs() || a->getNoImplicit() != b->getNoImplicit() ||
            a->getNumRadicalElectrons() != b->getNumRadicalElectrons() ||
            a->getProp<std::string>("_TriposAtomName") != atom_names[i] ||
            b->getProp<std::string>("_TriposAtomName") != atom_names[i])
            throw std::runtime_error("Coordinate neutralization changed atom identity/order");
        if (a->getAtomicNum() > 1) result.heavy_indices.push_back(i);
        if (b->getChiralTag() != RDKit::Atom::CHI_UNSPECIFIED) ++result.native_defined_stereo;
    }
    for (unsigned int i = 0; i < graph->getNumBonds(); ++i) {
        const auto* a = graph->getBondWithIdx(i);
        const auto* b = reference->getBondWithIdx(i);
        if (a->getBeginAtomIdx() != b->getBeginAtomIdx() || a->getEndAtomIdx() != b->getEndAtomIdx() ||
            a->getBondType() != b->getBondType() || a->getIsAromatic() != b->getIsAromatic())
            throw std::runtime_error("Coordinate neutralization changed bond graph/order");
        if (b->getStereo() != RDKit::Bond::STEREONONE) ++result.native_defined_stereo;
    }
    if (result.heavy_indices.empty()) throw std::invalid_argument("No heavy atoms in input");
    (void)heavy_coordinates(result.reference, result.heavy_indices);
    return result;
}

std::vector<std::array<double, 3>> heavy_coordinates(
    const RDKit::ROMol& molecule, const std::vector<unsigned int>& indices) {
    const auto& conformer = molecule.getConformer();
    std::vector<std::array<double, 3>> result;
    for (auto i : indices) {
        const auto& point = conformer.getAtomPos(i);
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            throw std::invalid_argument("Nonfinite evaluation coordinates");
        result.push_back({point.x, point.y, point.z});
    }
    return result;
}

} // namespace conformer_experiment
