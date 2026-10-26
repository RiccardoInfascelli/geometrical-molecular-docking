#ifndef CUDA_VINARDO_SCORING_HPP
#define CUDA_VINARDO_SCORING_HPP

#include "cuda_triplet_types.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace RDKit { class ROMol; }
class VinardoLikeScorer;
struct VinardoLikeScore;

struct GpuVinardoAtom {
    double x{}, y{}, z{}, radius{};
    std::uint32_t hydrophobic{}, donor{}, acceptor{};
    std::uint32_t padding{};
};

struct GpuVinardoAtomsDevice {
    GpuVinardoAtom* atoms{nullptr};
    std::uint32_t count{};
};

// Components already include their signed weights; lower totals are better.
// Repulsion is stored at alpha=1 so rep50 can be formed without rescoring pairs.
struct GpuVinardoScore {
    double total{}, gaussian{}, repulsion{}, hydrophobic{}, hydrogen_bond{};
    std::uint32_t valid{};
    std::uint32_t padding{};
};

struct GpuVinardoScoresDevice {
    GpuVinardoScore* scores{nullptr};
    std::uint32_t count{};
};

static_assert(std::is_trivially_copyable_v<GpuVinardoAtom>);
static_assert(std::is_trivially_copyable_v<GpuVinardoScore>);
static_assert(sizeof(GpuVinardoAtom) == 48);
static_assert(sizeof(GpuVinardoScore) == 48);

// Host typing uses the CPU scorer's RDKit rules, atom order and rejection rules.
std::vector<GpuVinardoAtom> make_gpu_vinardo_receptor_atoms(const VinardoLikeScorer& scorer);
std::vector<GpuVinardoAtom> make_gpu_vinardo_ligand_atoms(const RDKit::ROMol& ligand);
VinardoLikeScore vinardo_score_from_gpu(const GpuVinardoScore& score);

bool cuda_upload_vinardo_atoms(const std::vector<GpuVinardoAtom>& atoms,
                               GpuVinardoAtomsDevice& device);
void cuda_free_vinardo_atoms(GpuVinardoAtomsDevice& device);
void cuda_free_vinardo_scores(GpuVinardoScoresDevice& device);

// One thread scores each selected pose, retaining ligand/receptor accumulation
// order. Null IDs select every transform and require selected_count == count.
// IDs address resident clash-survivor transforms; output retains selection order.
// Base terms are unchanged: alpha is applied later without repeating atom pairs.
// The pair cutoff is a strict 8 A center distance, not a receptor crop.
// The blocking call waits for the kernel and marks nonfinite scores invalid.
bool cuda_score_vinardo_device(
    const GpuVinardoAtomsDevice& receptor,
    const GpuVinardoAtomsDevice& ligand,
    const GpuRigidTransformArrayDevice& transforms,
    const std::uint32_t* selected_pose_ids,
    std::uint32_t selected_count,
    GpuVinardoScoresDevice& output);

#endif
