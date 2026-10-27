#include "conformer_generation.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolAlign/AlignMolecules.h>
#include <GraphMol/MolOps.h>
#include <Geometry/Transform3D.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace conformer_experiment {
namespace {

constexpr const char* original_index = "_conformer_experiment_original_index";

void verify_graph_prefix(const RDKit::ROMol& source, const RDKit::ROMol& copy) {
    if (copy.getNumAtoms() < source.getNumAtoms() || copy.getNumBonds() < source.getNumBonds())
        throw std::runtime_error("Conformer generation removed original atoms or bonds");
    for (unsigned int i = 0; i < source.getNumAtoms(); ++i) {
        const auto* a = source.getAtomWithIdx(i);
        const auto* b = copy.getAtomWithIdx(i);
        if (a->getAtomicNum() != b->getAtomicNum() || a->getIsotope() != b->getIsotope() ||
            a->getFormalCharge() != b->getFormalCharge() || a->getChiralTag() != b->getChiralTag() ||
            a->getAtomMapNum() != b->getAtomMapNum() || a->getIsAromatic() != b->getIsAromatic() ||
            a->getNumRadicalElectrons() != b->getNumRadicalElectrons())
            throw std::runtime_error("Conformer generation changed original atom identity/stereo");
    }
    for (unsigned int i = 0; i < source.getNumBonds(); ++i) {
        const auto* a = source.getBondWithIdx(i);
        const auto* b = copy.getBondWithIdx(i);
        if (a->getBeginAtomIdx() != b->getBeginAtomIdx() || a->getEndAtomIdx() != b->getEndAtomIdx() ||
            a->getBondType() != b->getBondType() || a->getIsAromatic() != b->getIsAromatic() ||
            a->getStereo() != b->getStereo() || a->getBondDir() != b->getBondDir() ||
            a->getStereoAtoms() != b->getStereoAtoms())
            throw std::runtime_error("Conformer generation changed original bond graph/stereo");
    }
}

double aligned_rmsd(const RDKit::ROMol& work, int first, int second,
                    const RDKit::MatchVectType& heavy_atoms) {
    RDGeom::Transform3D transform;
    const double rmsd = RDKit::MolAlign::getAlignmentTransform(
        work, work, transform, first, second, &heavy_atoms, nullptr, false);
    if (!std::isfinite(rmsd))
        throw std::runtime_error("Conformer diversity RMSD is nonfinite");
    return rmsd;
}

void verify_coordinate_stereo(const RDKit::RWMol& reference,
                              const RDKit::ROMol& work, int conformer_id) {
    // Use the pre-embedding graph with identical explicit-H atom/neighbor order.
    // Perception is temporary: never assign unknown stereocenters to the output.
    RDKit::RWMol perceived(reference);
    perceived.clearConformers();
    perceived.addConformer(new RDKit::Conformer(work.getConformer(conformer_id)), true);
    RDKit::MolOps::assignStereochemistryFrom3D(perceived, -1, true);
    RDKit::MolOps::assignStereochemistry(perceived, true, true);
    for (unsigned int i = 0; i < reference.getNumAtoms(); ++i) {
        const auto expected = reference.getAtomWithIdx(i)->getChiralTag();
        if ((expected == RDKit::Atom::CHI_TETRAHEDRAL_CW || expected == RDKit::Atom::CHI_TETRAHEDRAL_CCW) &&
            perceived.getAtomWithIdx(i)->getChiralTag() != expected)
            throw std::runtime_error("Generated coordinates violate defined tetrahedral stereochemistry");
    }
    for (unsigned int i = 0; i < reference.getNumBonds(); ++i) {
        const auto expected = reference.getBondWithIdx(i)->getStereo();
        if ((expected == RDKit::Bond::STEREOE || expected == RDKit::Bond::STEREOZ) &&
            perceived.getBondWithIdx(i)->getStereo() != expected)
            throw std::runtime_error("Generated coordinates violate defined E/Z stereochemistry");
    }
}

} // namespace

ConformerEnsemble generate_conformers(const RDKit::RWMol& molecule,
                                     const ConformerOptions& options) {
    if (!options.count || options.count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        options.seed < 0 || !std::isfinite(options.min_rmsd) || options.min_rmsd < 0.0 ||
        options.max_iterations <= 0)
        throw std::invalid_argument("Invalid conformer count, seed, diversity threshold or iteration limit");
    if (!molecule.getNumAtoms() || molecule.getNumAtoms() > static_cast<unsigned int>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Conformer generation requires a nonempty atom-indexable molecule");
    for (const auto* atom : molecule.atoms()) {
        const auto tag = atom->getChiralTag();
        if (tag != RDKit::Atom::CHI_UNSPECIFIED && tag != RDKit::Atom::CHI_TETRAHEDRAL_CW &&
            tag != RDKit::Atom::CHI_TETRAHEDRAL_CCW)
            throw std::invalid_argument("Only tetrahedral atom stereochemistry is supported by this experiment");
    }
    for (const auto* bond : molecule.bonds()) {
        const auto stereo = bond->getStereo();
        if (stereo != RDKit::Bond::STEREONONE && stereo != RDKit::Bond::STEREOANY &&
            stereo != RDKit::Bond::STEREOE && stereo != RDKit::Bond::STEREOZ)
            throw std::invalid_argument("Only E/Z bond stereochemistry is supported by this experiment");
    }

    RDKit::MatchVectType heavy_atoms;
    RDKit::RWMol work(molecule);
    for (unsigned int i = 0; i < molecule.getNumAtoms(); ++i) {
        work.getAtomWithIdx(i)->setProp(original_index, i);
        if (molecule.getAtomWithIdx(i)->getAtomicNum() > 1)
            heavy_atoms.emplace_back(static_cast<int>(i), static_cast<int>(i));
    }
    if (heavy_atoms.empty())
        throw std::invalid_argument("Conformer diversity requires at least one heavy atom");
    if (work.getNumConformers() && work.getConformer().is3D()) {
        for (unsigned int i = 0; i < work.getNumAtoms(); ++i) {
            const auto& point = work.getConformer().getAtomPos(i);
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                throw std::invalid_argument("Input 3D conformer coordinates are nonfinite");
        }
        RDKit::RWMol defined_stereo(work);
        RDKit::MolOps::assignStereochemistry(defined_stereo, true, true);
        verify_coordinate_stereo(defined_stereo, work, -1);
        // MOL2 may encode stereo only through coordinates. Constrain embedding
        // from that geometry without adding any stereo tags to the output graph.
        RDKit::MolOps::assignStereochemistryFrom3D(work, -1, false);
    }
    work.clearConformers();
    const RDKit::RWMol constrained_graph(work);
    RDKit::MolOps::addHs(work);
    verify_graph_prefix(constrained_graph, work);
    RDKit::RWMol stereo_reference(work);
    RDKit::MolOps::assignStereochemistry(stereo_reference, true, true);

    auto parameters = RDKit::DGeomHelpers::ETKDGv3;
    parameters.randomSeed = options.seed;
    parameters.numThreads = 1;
    parameters.pruneRmsThresh = -1.0;
    parameters.enforceChirality = true;
    parameters.clearConfs = true;
    const auto ids = RDKit::DGeomHelpers::EmbedMultipleConfs(
        work, static_cast<unsigned int>(options.count), parameters);
    if (ids.empty())
        throw std::runtime_error("ETKDGv3 did not embed any conformers");
    verify_graph_prefix(constrained_graph, work);

    ConformerEnsemble result;
    result.requested = options.count;
    result.embedded = ids.size();
    result.mmff_fallback_reason = "not_requested";
    std::vector<std::pair<int, double>> optimized;
    if (options.optimize_mmff) {
        // Typing may change temporary aromaticity. Never invoke it in ETKDG-only
        // mode or export its graph: output always copies the untouched source.
        RDKit::MMFF::MMFFMolProperties properties(work, "MMFF94");
        result.mmff_supported = properties.isValid();
        result.mmff_used = result.mmff_supported;
        result.mmff_fallback_reason = result.mmff_supported ? "" : "unsupported_mmff94_parameters";
    }
    if (result.mmff_used) {
        RDKit::MMFF::MMFFOptimizeMoleculeConfs(work, optimized, 1, options.max_iterations, "MMFF94");
        if (optimized.size() != ids.size())
            throw std::runtime_error("MMFF94 result count differs from embedded conformer count");
    }
    struct Candidate { int id; std::optional<double> energy; };
    std::vector<Candidate> candidates;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (result.mmff_used && (optimized[i].first != 0 || !std::isfinite(optimized[i].second))) {
            ++result.mmff_failed;
            continue;
        }
        if (result.mmff_used) ++result.mmff_converged;
        candidates.push_back({ids[i], result.mmff_used ? std::optional<double>(optimized[i].second) : std::nullopt});
    }
    if (candidates.empty())
        throw std::runtime_error("No MMFF94 conformer optimization converged");
    if (result.mmff_used)
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return *a.energy < *b.energy || (*a.energy == *b.energy && a.id < b.id);
        });

    for (const auto& candidate : candidates) {
        std::optional<double> minimum;
        for (const auto& selected : result.conformers) {
            const double rmsd = aligned_rmsd(work, candidate.id, static_cast<int>(selected.id), heavy_atoms);
            minimum = minimum ? std::min(*minimum, rmsd) : rmsd;
        }
        if (minimum && *minimum < options.min_rmsd) continue;
        verify_coordinate_stereo(stereo_reference, work, candidate.id);
        GeneratedConformer selected;
        selected.id = static_cast<std::size_t>(candidate.id);
        selected.energy = candidate.energy;
        selected.minimum_aligned_rmsd = minimum;
        selected.molecule = molecule;
        selected.molecule.clearConformers();
        auto coordinates = std::make_unique<RDKit::Conformer>(molecule.getNumAtoms());
        coordinates->set3D(true);
        const auto& embedded = work.getConformer(candidate.id);
        for (unsigned int i = 0; i < molecule.getNumAtoms(); ++i) {
            if (work.getAtomWithIdx(i)->getProp<unsigned int>(original_index) != i)
                throw std::runtime_error("Conformer generation changed original atom mapping");
            const auto& point = embedded.getAtomPos(i);
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                throw std::runtime_error("Generated conformer coordinates are nonfinite");
            coordinates->setAtomPos(i, point);
        }
        selected.molecule.addConformer(coordinates.release(), true);
        verify_graph_prefix(molecule, selected.molecule);
        result.conformers.push_back(std::move(selected));
    }
    result.distinct = result.conformers.size();
    return result;
}

} // namespace conformer_experiment
