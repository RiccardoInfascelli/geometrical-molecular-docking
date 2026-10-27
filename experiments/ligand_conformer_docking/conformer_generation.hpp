#ifndef CONFORMER_EXPERIMENT_GENERATION_HPP
#define CONFORMER_EXPERIMENT_GENERATION_HPP

#include <GraphMol/RWMol.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace conformer_experiment {

struct ConformerOptions {
    std::size_t count = 10;
    int seed = 20261027;
    double min_rmsd = 0.5;
    bool optimize_mmff = false;
    int max_iterations = 500;
};

struct GeneratedConformer {
    // Original ETKDG candidate/conformer ID, retained after sorting/filtering.
    std::size_t id = 0;
    RDKit::RWMol molecule;
    // Isolated MMFF94 energy, present only after a converged optimization.
    // This energy never participates in docking ranking.
    std::optional<double> energy;
    // Heavy-atom, fixed-index aligned RMSD to the closest earlier selection.
    // No symmetry matching or comparison with input/crystal coordinates.
    std::optional<double> minimum_aligned_rmsd;
};

struct ConformerEnsemble {
    std::size_t requested = 0, embedded = 0, distinct = 0;
    // Support is checked only when optimization was requested; false otherwise.
    bool mmff_supported = false, mmff_used = false;
    std::size_t mmff_converged = 0, mmff_failed = 0;
    // "not_requested", "unsupported_mmff94_parameters", or empty when used.
    std::string mmff_fallback_reason;
    std::vector<GeneratedConformer> conformers;
};

// ETKDGv3: one thread, fixed nonnegative seed, internal pruning disabled.
// count is the number of candidates requested, not a promise of distinct output.
// MMFF94, when requested and supported, retains converged candidates in ascending
// energy/id order. Unsupported molecules retain ETKDG order and no energies.
// Greedy diversity filtering then uses min_rmsd (angstrom), without filling gaps.
// Only coordinates are copied onto the original atom graph; additional working hydrogens
// are never exported. Input atom identities, order, maps and stereo are retained.
// Existing 3D input supplies working-only stereo constraints even if its graph
// tags are unset. Defined graph stereo conflicting with those coordinates throws.
// Without usable 3D input, originally unspecified stereo stays unspecified.
// Defined tetrahedral and E/Z stereo is checked against the generated geometry;
// explicit stereo classes beyond tetrahedral atoms and E/Z bonds are rejected.
// Invalid options/input or absence of usable candidates throws std::exception.
ConformerEnsemble generate_conformers(const RDKit::RWMol& molecule,
                                     const ConformerOptions& options = {});

} // namespace conformer_experiment

#endif
