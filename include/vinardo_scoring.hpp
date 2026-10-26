#ifndef VINARDO_SCORING_HPP
#define VINARDO_SCORING_HPP

#include "geometry.hpp"

#include <GraphMol/ROMol.h>

#include <cstddef>
#include <vector>

// Vinardo-like prototype with approximate RDKit atom typing, not validated as
// AutoDock Vinardo. This intermolecular score compares poses of the same ligand;
// it is not a binding affinity in kcal/mol. Lower is better, unlike Fast/WeightedInteraction.
// Components already include coefficients and signs; total is their sum.
struct VinardoLikeScore {
    double total{};
    double gaussian{};
    double repulsion{};
    double hydrophobic{};
    double hydrogen_bond{};
};

// Atom typing is prepared on the host for both scoring backends. Coordinates
// retain their RDKit precision; ligand scoring rounds through coordinate_type.
struct VinardoAtomData {
    double x{}, y{}, z{};
    double radius{};
    bool hydrophobic{};
    bool donor{};
    bool acceptor{};
};

// Retain supported heavy atoms in RDKit order. Unsupported ligand elements
// throw, just as in the CPU scorer; no molecule pointers are retained.
std::vector<VinardoAtomData> prepare_vinardo_ligand_atoms(const RDKit::ROMol& ligand);

class VinardoLikeScorer {
public:
    // Copy supported heavy-atom data from the receptor conformer. Unsupported
    // receptor elements are skipped and counted; the molecule need not outlive us.
    explicit VinardoLikeScorer(const RDKit::ROMol& receptor);

    // Transform the ligand conformer and sum heavy-atom pairs at distance < 8 A.
    // Missing conformers or unsupported ligand elements throw. No intramolecular
    // or torsional term is included, and the input molecule is left unchanged.
    VinardoLikeScore score(
        const RDKit::ROMol& ligand,
        const RigidTransform& transform) const;

    std::size_t receptor_skipped_unsupported() const {
        return skipped_receptor_atoms_;
    }

    const std::vector<VinardoAtomData>& receptor_atoms() const {
        return receptor_atoms_;
    }

private:
    std::vector<VinardoAtomData> receptor_atoms_;
    std::size_t skipped_receptor_atoms_{};
};

#endif
