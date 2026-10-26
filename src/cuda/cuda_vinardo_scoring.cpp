#include "cuda_vinardo_scoring.hpp"

#include "vinardo_scoring.hpp"

#include <cmath>
#include <stdexcept>

namespace {

static_assert(std::is_same_v<interaction::coordinate_type, float>,
              "CUDA Vinardo must follow the CPU coordinate narrowing");

std::vector<GpuVinardoAtom> pack_atoms(const std::vector<VinardoAtomData>& atoms) {
    std::vector<GpuVinardoAtom> packed;
    packed.reserve(atoms.size());
    for (const auto& atom : atoms) {
        packed.push_back({atom.x, atom.y, atom.z, atom.radius,
                          static_cast<std::uint32_t>(atom.hydrophobic),
                          static_cast<std::uint32_t>(atom.donor),
                          static_cast<std::uint32_t>(atom.acceptor), 0});
    }
    return packed;
}

}

std::vector<GpuVinardoAtom> make_gpu_vinardo_receptor_atoms(const VinardoLikeScorer& scorer) {
    return pack_atoms(scorer.receptor_atoms());
}

std::vector<GpuVinardoAtom> make_gpu_vinardo_ligand_atoms(const RDKit::ROMol& ligand) {
    return pack_atoms(prepare_vinardo_ligand_atoms(ligand));
}

VinardoLikeScore vinardo_score_from_gpu(const GpuVinardoScore& score) {
    if (!score.valid || !std::isfinite(score.total) || !std::isfinite(score.gaussian) ||
        !std::isfinite(score.repulsion) || !std::isfinite(score.hydrophobic) ||
        !std::isfinite(score.hydrogen_bond))
        throw std::runtime_error("Nonfinite CUDA Vinardo-like score");
    return {score.total, score.gaussian, score.repulsion,
            score.hydrophobic, score.hydrogen_bond};
}
