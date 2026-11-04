#include "conformer_generation.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/MolAlign/AlignMolecules.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <Geometry/Transform3D.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

using namespace conformer_experiment;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::unique_ptr<RDKit::RWMol> molecule(const std::string& smiles) {
    std::unique_ptr<RDKit::RWMol> result(RDKit::SmilesToMol(smiles));
    require(result != nullptr, "Invalid synthetic fixture");
    return result;
}
void same_graph(const RDKit::RWMol& source, const RDKit::RWMol& generated) {
    require(source.getNumAtoms() == generated.getNumAtoms() &&
            source.getNumBonds() == generated.getNumBonds(), "Atom/bond counts changed");
    for (unsigned int i = 0; i < source.getNumAtoms(); ++i) {
        const auto* a = source.getAtomWithIdx(i);
        const auto* b = generated.getAtomWithIdx(i);
        require(a->getAtomicNum() == b->getAtomicNum() && a->getIsotope() == b->getIsotope() &&
                a->getFormalCharge() == b->getFormalCharge() && a->getChiralTag() == b->getChiralTag() &&
                a->getAtomMapNum() == b->getAtomMapNum() && a->getIsAromatic() == b->getIsAromatic() &&
                a->getNumExplicitHs() == b->getNumExplicitHs() && a->getNoImplicit() == b->getNoImplicit(),
                "Atom identity/order/stereo changed");
        require(a->getProp<std::string>("_TriposAtomName") == b->getProp<std::string>("_TriposAtomName"),
                "Original atom names changed");
    }
    for (unsigned int i = 0; i < source.getNumBonds(); ++i) {
        const auto* a = source.getBondWithIdx(i);
        const auto* b = generated.getBondWithIdx(i);
        require(a->getBeginAtomIdx() == b->getBeginAtomIdx() && a->getEndAtomIdx() == b->getEndAtomIdx() &&
                a->getBondType() == b->getBondType() && a->getIsAromatic() == b->getIsAromatic() &&
                a->getStereo() == b->getStereo() && a->getStereoAtoms() == b->getStereoAtoms() &&
                a->getBondDir() == b->getBondDir(), "Bond graph/stereo changed");
    }
}
void test_identity_and_seed() {
    for (bool explicit_hydrogens : {false, true}) {
        auto input = molecule("F[C@H](Cl)/C=C/[CH2:7][OH:8]");
        if (explicit_hydrogens) RDKit::MolOps::addHs(*input);
        for (auto* atom : input->atoms())
            atom->setProp("_TriposAtomName", std::string("atom_") + std::to_string(atom->getIdx()));
        ConformerOptions options;
        options.count = 3; options.min_rmsd = 0.0;
        const auto same_generation = [&](const ConformerEnsemble& first, const ConformerEnsemble& second,
                                         const char* coordinate_error) {
            require(first.requested == 3 && second.requested == 3 && first.embedded == 3 && second.embedded == 3 &&
                    first.distinct == 3 && second.distinct == 3 && first.conformers.size() == second.conformers.size(),
                    "Seeded ETKDG candidate count differs");
            for (std::size_t c = 0; c < first.conformers.size(); ++c) {
                const auto& a = first.conformers[c];
                const auto& b = second.conformers[c];
                require(a.id == b.id && !a.energy && !b.energy && !first.mmff_used && !second.mmff_used &&
                        !first.mmff_supported && !second.mmff_supported &&
                        first.mmff_fallback_reason == "not_requested" && second.mmff_fallback_reason == "not_requested" &&
                        a.molecule.getNumConformers() == 1 && b.molecule.getNumConformers() == 1,
                        "ETKDG order/metadata differs");
                same_graph(*input, a.molecule);
                same_graph(*input, b.molecule);
                for (unsigned int i = 0; i < input->getNumAtoms(); ++i) {
                    const auto& x = a.molecule.getConformer().getAtomPos(i);
                    const auto& y = b.molecule.getConformer().getAtomPos(i);
                    require(std::bit_cast<std::uint64_t>(x.x) == std::bit_cast<std::uint64_t>(y.x) &&
                            std::bit_cast<std::uint64_t>(x.y) == std::bit_cast<std::uint64_t>(y.y) &&
                            std::bit_cast<std::uint64_t>(x.z) == std::bit_cast<std::uint64_t>(y.z), coordinate_error);
                }
            }
        };
        const auto first = generate_conformers(*input, options);
        const auto second = generate_conformers(*input, options);
        require(input->getNumConformers() == 0, "Generation mutated input conformers");
        same_generation(first, second, "Seeded generation is not bitwise repeatable without 3D");
        auto* original = new RDKit::Conformer(first.conformers.front().molecule.getConformer());
        const double source_x = original->getAtomPos(1).x;
        input->addConformer(original, true);
        const auto third = generate_conformers(*input, options);
        const auto fourth = generate_conformers(*input, options);
        same_generation(third, fourth, "Seeded generation is not bitwise repeatable with 3D");
        require(input->getNumConformers() == 1 && input->getConformer().getAtomPos(1).x == source_x,
                "Generation changed original coordinates");
        for (unsigned int i = 0; i < input->getNumAtoms(); ++i)
            input->getConformer().setAtomPos(i, input->getConformer().getAtomPos(i) + RDGeom::Point3D(100.0, 0.0, 0.0));
        const double translated_x = input->getConformer().getAtomPos(1).x;
        const auto translated = generate_conformers(*input, options);
        same_generation(third, translated, "Generated coordinates depend on translation of input 3D");
        require(input->getNumConformers() == 1 && input->getConformer().getAtomPos(1).x == translated_x,
                "Generation changed translated input coordinates");
    }
}
void test_diversity() {
    auto input = molecule("CCCCCC");
    ConformerOptions options;
    options.count = 12; options.min_rmsd = 0.6;
    const auto ensemble = generate_conformers(*input, options);
    require(ensemble.requested == 12 && ensemble.embedded <= 12 && ensemble.distinct == ensemble.conformers.size() &&
            ensemble.distinct >= 1 && ensemble.distinct <= ensemble.embedded, "Diversity counts are inconsistent");
    RDKit::MatchVectType atom_map;
    for (unsigned int i = 0; i < input->getNumAtoms(); ++i) atom_map.emplace_back(i, i);
    for (std::size_t i = 0; i < ensemble.conformers.size(); ++i) {
        std::optional<double> minimum;
        for (std::size_t j = 0; j < i; ++j) {
            RDGeom::Transform3D transform;
            const double rmsd = RDKit::MolAlign::getAlignmentTransform(ensemble.conformers[i].molecule,
                ensemble.conformers[j].molecule, transform, -1, -1, &atom_map, nullptr, false);
            require(rmsd >= options.min_rmsd - 1e-9, "Returned conformers violate diversity threshold");
            minimum = minimum ? std::min(*minimum, rmsd) : rmsd;
        }
        require(minimum.has_value() == ensemble.conformers[i].minimum_aligned_rmsd.has_value(),
                "First conformer should have no earlier diversity RMSD");
        if (minimum) require(std::abs(*minimum - *ensemble.conformers[i].minimum_aligned_rmsd) < 1e-8,
                             "Reported diversity RMSD differs from fixed-index alignment");
    }
    options.count = 4; options.min_rmsd = 1e6;
    require(generate_conformers(*input, options).distinct == 1, "Duplicate slots were filled or threshold relaxed");
}
void test_coordinate_stereo() {
    auto input = molecule("F[C@H](Cl)/C=C/[CH2:7][OH:8]");
    for (bool mmff : {false, true}) {
        ConformerOptions options;
        options.count = 2; options.min_rmsd = 0.0; options.optimize_mmff = mmff;
        const auto ensemble = generate_conformers(*input, options);
        for (const auto& conformer : ensemble.conformers) {
            RDKit::RWMol perceived(conformer.molecule);
            RDKit::MolOps::assignStereochemistryFrom3D(perceived, -1, true);
            RDKit::MolOps::assignStereochemistry(perceived, true, true);
            bool tested_center = false, tested_bond = false;
            for (const auto* atom : input->atoms()) {
                const auto expected = atom->getChiralTag();
                if (expected == RDKit::Atom::CHI_TETRAHEDRAL_CW || expected == RDKit::Atom::CHI_TETRAHEDRAL_CCW) {
                    tested_center = true;
                    require(perceived.getAtomWithIdx(atom->getIdx())->getChiralTag() == expected,
                            "Generated coordinates inverted a defined stereocenter");
                }
            }
            for (const auto* bond : input->bonds()) {
                const auto expected = bond->getStereo();
                if (expected == RDKit::Bond::STEREOE || expected == RDKit::Bond::STEREOZ) {
                    tested_bond = true;
                    require(perceived.getBondWithIdx(bond->getIdx())->getStereo() == expected,
                            "Generated coordinates inverted defined E/Z stereochemistry");
                }
            }
            require(tested_center && tested_bond, "Stereo fixture lacks defined tetrahedral/E/Z geometry");
        }
        // Emulate a MOL2 parser preserving 3D coordinates but leaving tags unset.
        RDKit::RWMol untagged(ensemble.conformers.front().molecule);
        for (auto* atom : untagged.atoms()) {
            atom->setChiralTag(RDKit::Atom::CHI_UNSPECIFIED);
            if (atom->hasProp("_CIPCode")) atom->clearProp("_CIPCode");
        }
        for (auto* bond : untagged.bonds()) {
            bond->setStereo(RDKit::Bond::STEREONONE);
            bond->setBondDir(RDKit::Bond::NONE);
            bond->getStereoAtoms().clear();
        }
        const auto from_coordinates = generate_conformers(untagged, options);
        for (const auto& conformer : from_coordinates.conformers) {
            require(conformer.molecule.getAtomWithIdx(1)->getChiralTag() == RDKit::Atom::CHI_UNSPECIFIED,
                    "Working-only 3D inference changed the output graph");
            RDKit::RWMol perceived(conformer.molecule);
            RDKit::MolOps::assignStereochemistryFrom3D(perceived, -1, true);
            RDKit::MolOps::assignStereochemistry(perceived, true, true);
            require(perceived.getAtomWithIdx(1)->getChiralTag() == input->getAtomWithIdx(1)->getChiralTag(),
                    "Tag-unset input 3D chirality was randomized");
            for (const auto* bond : input->bonds())
                if (bond->getStereo() == RDKit::Bond::STEREOE || bond->getStereo() == RDKit::Bond::STEREOZ)
                    require(perceived.getBondWithIdx(bond->getIdx())->getStereo() == bond->getStereo(),
                            "Tag-unset input 3D E/Z geometry was randomized");
        }
        // A mirror retains copied graph tags but must reverse perceived chirality.
        RDKit::RWMol mirrored(ensemble.conformers.front().molecule);
        for (unsigned int i = 0; i < mirrored.getNumAtoms(); ++i) {
            auto point = mirrored.getConformer().getAtomPos(i);
            point.x = -point.x;
            mirrored.getConformer().setAtomPos(i, point);
        }
        bool rejected_conflict = false;
        try { (void)generate_conformers(mirrored, options); }
        catch (const std::runtime_error&) { rejected_conflict = true; }
        require(rejected_conflict, "Input 3D contradicting defined stereo was accepted");
        RDKit::MolOps::assignStereochemistryFrom3D(mirrored, -1, true);
        require(mirrored.getAtomWithIdx(1)->getChiralTag() != input->getAtomWithIdx(1)->getChiralTag() &&
                mirrored.getAtomWithIdx(1)->getChiralTag() != RDKit::Atom::CHI_UNSPECIFIED,
                "Geometry check merely reused copied stereochemical tags");
    }
    auto unassigned = molecule("FC(Cl)(Br)I");
    ConformerOptions one;
    one.count = 1;
    const auto unknown = generate_conformers(*unassigned, one);
    require(unknown.conformers.front().molecule.getAtomWithIdx(1)->getChiralTag() == RDKit::Atom::CHI_UNSPECIFIED,
            "Generation assigned originally unknown stereochemistry");
}
void test_mmff() {
    ConformerOptions options;
    options.count = 3; options.min_rmsd = 0.0; options.optimize_mmff = true;
    const auto optimized = generate_conformers(*molecule("CCCC"), options);
    require(optimized.mmff_supported && optimized.mmff_used && optimized.mmff_converged > 0 &&
            optimized.mmff_converged + optimized.mmff_failed == optimized.embedded &&
            optimized.distinct == optimized.mmff_converged && optimized.mmff_fallback_reason.empty(),
            "MMFF convergence metadata differs");
    double previous = -std::numeric_limits<double>::infinity();
    for (const auto& conformer : optimized.conformers) {
        require(conformer.energy && std::isfinite(*conformer.energy) && *conformer.energy >= previous,
                "Converged MMFF energies are missing or unsorted");
        previous = *conformer.energy;
    }
    const auto fallback = generate_conformers(*molecule("B(O)(O)O"), options);
    require(!fallback.mmff_supported && !fallback.mmff_used && !fallback.mmff_converged && !fallback.mmff_failed &&
            fallback.mmff_fallback_reason == "unsupported_mmff94_parameters", "Unsupported MMFF fallback differs");
    for (std::size_t i = 0; i < fallback.conformers.size(); ++i)
        require(!fallback.conformers[i].energy && (i == 0 || fallback.conformers[i - 1].id < fallback.conformers[i].id),
                "Unsupported MMFF introduced energy ranking");
}
void test_invalid() {
    auto input = molecule("CC");
    const auto rejects = [&](const RDKit::RWMol& mol, const ConformerOptions& options) {
        try { (void)generate_conformers(mol, options); }
        catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("Invalid input/options were accepted");
    };
    ConformerOptions options;
    options.count = 0; rejects(*input, options);
    options = {}; options.count = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1; rejects(*input, options);
    options = {}; options.seed = -1; rejects(*input, options);
    options = {}; options.max_iterations = 0; rejects(*input, options);
    for (double value : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        options = {}; options.min_rmsd = value; rejects(*input, options);
    }
    rejects(RDKit::RWMol{}, {});
    rejects(*molecule("[H][H]"), {});
    auto unsupported = molecule("CC");
    unsupported->getAtomWithIdx(0)->setChiralTag(RDKit::Atom::CHI_OTHER);
    rejects(*unsupported, {});
}
} // namespace

int main() {
    try {
        test_identity_and_seed(); test_diversity(); test_coordinate_stereo(); test_mmff(); test_invalid();
        std::cout << "Conformer generation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Conformer generation test failed: " << error.what() << '\n';
        return 1;
    }
}
