#include "vinardo_scoring.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/ROMol.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

struct AtomType {
    double radius{};
    bool hydrophobic{};
    bool donor{};
    bool acceptor{};
};

// Use Vinardo radii for C/C_ar/N/O and estimates for the other supported elements.
// Donor, acceptor and hydrophobic flags approximate typing through RDKit.
std::optional<AtomType> classify(const RDKit::Atom& atom) {
    const auto z = atom.getAtomicNum();
    if (z == 1) return std::nullopt;   // Hydrogens are excluded from pair scoring.

    AtomType t;
    switch (z) {
    case 6:  t.radius = atom.getIsAromatic() ? 1.9 : 2.0; break;
    case 7:  t.radius = 1.7; break;
    case 8:  t.radius = 1.6; break;
    case 16: t.radius = 2.0; break;
    case 15: t.radius = 2.1; break;
    case 9:  t.radius = 1.8; break;
    case 17: t.radius = 1.8; break;
    case 35: t.radius = 2.0; break;
    case 53: t.radius = 2.2; break;
    default: return std::nullopt;
    }

    if (z == 6) {
        // Carbon bonded directly to N/O is not treated as hydrophobic.
        t.hydrophobic = true;
        for (const auto* neighbor : atom.getOwningMol().atomNeighbors(&atom)) {
            const auto zn = neighbor->getAtomicNum();
            if (zn == 7 || zn == 8) {
                t.hydrophobic = false;
                break;
            }
        }
    } else if (z == 16 || z == 9 || z == 17 || z == 35 || z == 53) {
        t.hydrophobic = true;
    }

    if (z == 7 || z == 8) {
        const bool positive = atom.getFormalCharge() > 0;
        // These simple donor/acceptor rules are not XS/PDBQT typing.
        t.donor = (z == 7 && !atom.getIsAromatic()) ||
                  (atom.getTotalNumHs() > 0);
        t.acceptor = !positive && (z == 8 ||
                     (z == 7 && atom.getTotalNumHs() == 0));
    }
    return t;
}

double hbond_ramp(double d) {
    if (d <= -0.6) return 1.0;
    if (d >= 0.0) return 0.0;
    return -d / 0.6;
}

double hydrophobic_ramp(double d) {
    if (d <= 0.0) return 1.0;
    if (d >= 2.5) return 0.0;
    return 1.0 - d / 2.5;
}

}  // namespace

std::vector<VinardoAtomData> prepare_vinardo_ligand_atoms(const RDKit::ROMol& ligand) {
    if (!ligand.getNumConformers())
        throw std::runtime_error("Ligando senza conformero 3D");

    std::vector<VinardoAtomData> atoms;
    atoms.reserve(ligand.getNumAtoms());
    const auto& conformer = ligand.getConformer();
    for (const auto* atom : ligand.atoms()) {
        if (atom->getAtomicNum() == 1) continue;
        const auto type = classify(*atom);
        if (!type) {
            throw std::runtime_error(
                "Elemento ligando non supportato, numero atomico " +
                std::to_string(atom->getAtomicNum()));
        }
        const auto p = conformer.getAtomPos(atom->getIdx());
        atoms.push_back({p.x, p.y, p.z, type->radius,
                         type->hydrophobic, type->donor, type->acceptor});
    }
    return atoms;
}

VinardoLikeScorer::VinardoLikeScorer(const RDKit::ROMol& receptor) {
    if (!receptor.getNumConformers())
        throw std::runtime_error("Recettore senza conformero 3D");

    const auto& conformer = receptor.getConformer();
    for (const auto* atom : receptor.atoms()) {
        if (atom->getAtomicNum() == 1) continue;
        const auto type = classify(*atom);
        if (!type) {
            ++skipped_receptor_atoms_;
            continue;
        }
        const auto p = conformer.getAtomPos(atom->getIdx());
        receptor_atoms_.push_back({
            p.x, p.y, p.z, type->radius,
            type->hydrophobic, type->donor, type->acceptor
        });
    }
}

VinardoLikeScore VinardoLikeScorer::score(
    const RDKit::ROMol& ligand,
    const RigidTransform& transform) const
{
    if (!ligand.getNumConformers())
        throw std::runtime_error("Ligando senza conformero 3D");

    VinardoLikeScore result;
    const auto& conf = ligand.getConformer();

    for (const auto* atom : ligand.atoms()) {
        if (atom->getAtomicNum() == 1) continue;
        const auto type = classify(*atom);
        if (!type) {
            throw std::runtime_error(
                "Elemento ligando non supportato, numero atomico " +
                std::to_string(atom->getAtomicNum()));
        }
        const auto original = conf.getAtomPos(atom->getIdx());
        // Preserve coordinate_type rounding both before and after the transform.
        const interaction::point3D source{
            static_cast<interaction::coordinate_type>(original.x),
            static_cast<interaction::coordinate_type>(original.y),
            static_cast<interaction::coordinate_type>(original.z)
        };
        const auto p = transform_point(source, transform);

        for (const auto& r : receptor_atoms_) {
            const double dx = static_cast<double>(p.x) - r.x;
            const double dy = static_cast<double>(p.y) - r.y;
            const double dz = static_cast<double>(p.z) - r.z;
            const double squared_distance = dx*dx + dy*dy + dz*dz;
            if (squared_distance >= 64.0) continue;  // Strict 8 A pair cutoff.

            const double dist = std::sqrt(squared_distance);
            // Surface separation: negative values mean overlapping atomic radii.
            const double d = dist - (type->radius + r.radius);
            const double x = d / 0.8;
            result.gaussian += -0.045 * std::exp(-(x*x));
            if (d < 0.0) result.repulsion += 0.8 * d*d;
            if (type->hydrophobic && r.hydrophobic)
                result.hydrophobic += -0.035 * hydrophobic_ramp(d);
            if ((type->donor && r.acceptor) ||
                (type->acceptor && r.donor))
                result.hydrogen_bond += -0.6 * hbond_ramp(d);
        }
    }

    result.total = result.gaussian + result.repulsion +
                   result.hydrophobic + result.hydrogen_bond;
    return result;
}
