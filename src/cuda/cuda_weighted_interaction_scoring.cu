#include "cuda_weighted_interaction_scoring.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace {

using Point = GpuWeightedInteractionPoint;
using Pattern = GpuWeightedInteractionPattern;

__device__ Point difference(Point a, Point b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ Point opposite(Point a) { return {-a.x, -a.y, -a.z}; }
__device__ Point multiply(Point a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ float dot(Point a, Point b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float distance2(Point a, Point b) {
    const auto delta = difference(a, b);
    return dot(delta, delta);
}
__device__ Point normal(const GpuWeightedInteractionArea& area) {
    const auto a = difference(area.atoms[0], area.centroid);
    const auto b = difference(area.atoms[1], area.centroid);
    // Preserve chem-interactions' y-component sign, rather than changing its
    // plane convention while porting the predicates to CUDA.
    return {a.y * b.z - a.z * b.y, a.x * b.z - a.z * b.x,
            a.x * b.y - a.y * b.x};
}
__device__ Point normalize(Point point) {
    const float norm = sqrtf(dot(point, point));
    return {point.x / norm, point.y / norm, point.z / norm};
}
__device__ float angle(Point a, Point b) {
    // No clamp or degeneracy epsilon: the CPU dependency also retains acos NaNs.
    return acosf(dot(a, b) / sqrtf(dot(a, a) * dot(b, b)));
}
__device__ float radians(float degrees) {
    return degrees * (atanf(1.0f) * 4.0f) / 180.0f;
}
__device__ float minimum(float a, float b) {
    // Match std::min's ordered comparison, including NaN behavior.
    return b < a ? b : a;
}
__device__ bool within(float value, float low, float high) {
    return value >= radians(low) && value <= radians(high);
}
__device__ Point transform_point(Point point, const GpuRigidTransform& transform) {
    const double x = point.x, y = point.y, z = point.z;
    // The CPU transformation is evaluated in double and stored back as float.
    return {
        static_cast<float>(transform.rotation[0] * x + transform.rotation[1] * y +
                           transform.rotation[2] * z + transform.translation[0]),
        static_cast<float>(transform.rotation[3] * x + transform.rotation[4] * y +
                           transform.rotation[5] * z + transform.translation[1]),
        static_cast<float>(transform.rotation[6] * x + transform.rotation[7] * y +
                           transform.rotation[8] * z + transform.translation[2])};
}
__device__ GpuWeightedInteractionArea transform_area(GpuWeightedInteractionArea area, const GpuRigidTransform& transform) {
    area.centroid = transform_point(area.centroid, transform);
    for (std::uint32_t i = 0; i < 2 && i < area.num_atoms; ++i)
        area.atoms[i] = transform_point(area.atoms[i], transform);
    return area;
}

__device__ bool hydrophobic(const GpuWeightedInteractionArea& a, const GpuWeightedInteractionArea& b) {
    return a.pattern == Pattern::Hydrophobic && b.pattern == Pattern::Hydrophobic &&
           distance2(a.centroid, b.centroid) <= 4.5f * 4.5f;
}

__device__ bool hydrogen_bond(const GpuWeightedInteractionArea& a, const GpuWeightedInteractionArea& b) {
    const GpuWeightedInteractionArea* donor = nullptr;
    const GpuWeightedInteractionArea* acceptor = nullptr;
    if (a.pattern == Pattern::HydrogenDonor && b.pattern == Pattern::HydrogenAcceptor) {
        donor = &a; acceptor = &b;
    } else if (b.pattern == Pattern::HydrogenDonor && a.pattern == Pattern::HydrogenAcceptor) {
        donor = &b; acceptor = &a;
    } else return false;
    if (!(distance2(donor->centroid, acceptor->centroid) <= 3.5f * 3.5f)) return false;
    return within(angle(difference(donor->atoms[0], donor->atoms[1]),
                        difference(acceptor->atoms[0], donor->atoms[1])), 130.0f, 180.0f);
}

__device__ bool halogen_bond(const GpuWeightedInteractionArea& a, const GpuWeightedInteractionArea& b) {
    const GpuWeightedInteractionArea* donor = nullptr;
    const GpuWeightedInteractionArea* acceptor = nullptr;
    if (a.pattern == Pattern::HalogenDonor && b.pattern == Pattern::HalogenAcceptor) {
        donor = &a; acceptor = &b;
    } else if (b.pattern == Pattern::HalogenDonor && a.pattern == Pattern::HalogenAcceptor) {
        donor = &b; acceptor = &a;
    } else return false;
    if (!(distance2(donor->centroid, acceptor->centroid) <= 3.5f * 3.5f)) return false;
    const Point h = donor->atoms[1], acceptor_atom = acceptor->atoms[0];
    if (!within(angle(difference(donor->atoms[0], h), difference(acceptor_atom, h)),
                130.0f, 180.0f)) return false;
    return within(angle(difference(h, acceptor_atom),
                        difference(acceptor->atoms[1], acceptor_atom)), 80.0f, 140.0f);
}

__device__ bool cation_pi_pair(const GpuWeightedInteractionArea& a, const GpuWeightedInteractionArea& b) {
    return (a.aromatic && b.pattern == Pattern::Cation) ||
           (b.aromatic && a.pattern == Pattern::Cation);
}

__device__ bool ionic(const GpuWeightedInteractionArea& a, const GpuWeightedInteractionArea& b) {
    if (((a.pattern == Pattern::Anion && b.pattern == Pattern::Cation) ||
         (b.pattern == Pattern::Anion && a.pattern == Pattern::Cation)) &&
        distance2(a.centroid, b.centroid) <= 4.5f * 4.5f) return true;
    const GpuWeightedInteractionArea* aromatic = nullptr;
    const GpuWeightedInteractionArea* cation = nullptr;
    if (a.aromatic && b.pattern == Pattern::Cation) { aromatic = &a; cation = &b; }
    else if (b.aromatic && a.pattern == Pattern::Cation) { aromatic = &b; cation = &a; }
    else return false;
    if (!(distance2(aromatic->centroid, cation->centroid) <= 4.5f * 4.5f)) return false;
    const auto n = normal(*aromatic);
    const auto cc = difference(cation->centroid, aromatic->centroid);
    return within(minimum(angle(n, cc), angle(opposite(n), cc)), 0.0f, 30.0f);
}

__device__ bool face_to_face(const GpuWeightedInteractionArea& a, const GpuWeightedInteractionArea& b) {
    if (!a.aromatic || !b.aromatic ||
        !(distance2(a.centroid, b.centroid) <= 5.5f * 5.5f)) return false;
    const auto normal1 = normal(a), normal2 = normal(b);
    const auto cc12 = difference(b.centroid, a.centroid), cc21 = opposite(cc12);
    const float angle11 = angle(cc12, normal1), angle12 = angle(cc12, opposite(normal1));
    const float angle21 = angle(cc21, normal2), angle22 = angle(cc21, opposite(normal2));
    const auto n1 = angle11 <= angle12 ? normal1 : opposite(normal1);
    const auto n2 = angle21 <= angle22 ? normal2 : opposite(normal2);
    const float aligned1 = minimum(angle11, angle12), aligned2 = minimum(angle21, angle22);
    if (aligned1 < radians(0.0f) || aligned1 > radians(33.0f)) return false;
    if (aligned2 < radians(0.0f) || aligned2 > radians(33.0f)) return false;
    return within(angle(n1, n2), 0.0f, 30.0f);
}

__device__ bool edge_to_face(const GpuWeightedInteractionArea& a, const GpuWeightedInteractionArea& b) {
    if (!a.aromatic || !b.aromatic ||
        !(distance2(a.centroid, b.centroid) <= 6.5f * 6.5f)) return false;
    const auto normal1 = normal(a), normal2 = normal(b);
    const auto cc12 = difference(b.centroid, a.centroid), cc21 = opposite(cc12);
    const float angle11 = angle(cc12, normal1), angle12 = angle(cc12, opposite(normal1));
    const float angle21 = angle(cc21, normal2), angle22 = angle(cc21, opposite(normal2));
    const auto n1 = angle11 <= angle12 ? normal1 : opposite(normal1);
    const auto n2 = angle21 <= angle22 ? normal2 : opposite(normal2);
    const float aligned1 = minimum(angle11, angle12), aligned2 = minimum(angle21, angle22);
    if (aligned1 < radians(0.0f) || aligned1 > radians(33.0f)) return false;
    if (aligned2 < radians(0.0f) || aligned2 > radians(33.0f)) return false;
    const float plane = angle(n1, n2);
    if (plane < radians(50.0f) || plane > radians(90.0f)) return false;
    const float distance = sqrtf(distance2(a.centroid, b.centroid));
    const auto projected2 = difference(b.centroid, multiply(normalize(n1), distance));
    const auto projected1 = difference(a.centroid, multiply(normalize(n2), distance));
    return minimum(distance2(a.centroid, projected2), distance2(b.centroid, projected1)) <=
           1.7f * 1.7f;
}

__device__ bool metal(const GpuWeightedInteractionArea& a, const GpuWeightedInteractionArea& b) {
    return ((a.pattern == Pattern::Metal && b.pattern == Pattern::Chelated) ||
            (b.pattern == Pattern::Metal && a.pattern == Pattern::Chelated)) &&
           distance2(a.centroid, b.centroid) <= 3.5f * 3.5f;
}

__device__ void record_contact(bool formed, std::uint64_t& count,
                               double weight, bool& ligand_satisfied,
                               unsigned char& protein_satisfied, double& best,
                               GpuWeightedInteractionScore& score) {
    if (!formed) return;
    ++count;
    ++score.total_interactions;
    ligand_satisfied = true;
    protein_satisfied = 1;
    if (best < weight) best = weight;
}

__global__ void weighted_interaction_score_kernel(
    GpuWeightedInteractionAreasDevice protein, GpuWeightedInteractionAreasDevice ligand,
    const GpuRigidTransform* transforms, std::uint32_t transform_count,
    const std::uint32_t* selected, std::uint32_t count,
    GpuWeightedInteractionWeights weights, unsigned char* satisfied_protein,
    GpuWeightedInteractionScore* output) {
    const auto linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= count) return;
    const auto index = static_cast<std::uint32_t>(linear);
    GpuWeightedInteractionScore score{};
    const std::uint32_t pose_id = selected ? selected[index] : index;
    if (pose_id >= transform_count || !transforms[pose_id].valid) {
        output[index] = score;
        return;
    }
    const auto transform = transforms[pose_id];
    auto* coverage = protein.count ? satisfied_protein +
        static_cast<std::size_t>(index) * protein.count : nullptr;
    for (std::uint32_t p = 0; p < protein.count; ++p) coverage[p] = 0;
    for (std::uint32_t l = 0; l < ligand.count; ++l) {
        const auto ligand_area = transform_area(ligand.areas[l], transform);
        bool satisfied = false;
        // The reward belongs to this physical area, not to each contact.
        // Starting at zero also preserves CPU behavior for negative weights.
        double best = 0.0;
        // The current CPU dependency leaves refine_interaction_set() disabled:
        // all formed contacts contribute, with no atom-level dominance filter.
        for (std::uint32_t p = 0; p < protein.count; ++p) {
            const auto& protein_area = protein.areas[p];
            // Keep ligand first, matching find_interactions() even for NaNs.
            record_contact(hydrophobic(ligand_area, protein_area), score.hydrophobic,
                           weights.hydrophobic, satisfied, coverage[p], best, score);
            record_contact(hydrogen_bond(ligand_area, protein_area), score.hydrogen_bonds,
                           weights.hydrogen_bond, satisfied, coverage[p], best, score);
            record_contact(halogen_bond(ligand_area, protein_area), score.halogen_bonds,
                           weights.halogen_bond, satisfied, coverage[p], best, score);
            const bool cation_pi = cation_pi_pair(ligand_area, protein_area);
            record_contact(ionic(ligand_area, protein_area),
                           cation_pi ? score.cation_pi : score.ionic,
                           cation_pi ? weights.cation_pi : weights.ionic,
                           satisfied, coverage[p], best, score);
            record_contact(face_to_face(ligand_area, protein_area) ||
                           edge_to_face(ligand_area, protein_area), score.pi_stacking,
                           weights.pi_stacking, satisfied, coverage[p], best, score);
            record_contact(metal(ligand_area, protein_area), score.metal,
                           weights.metal, satisfied, coverage[p], best, score);
            // Vdw_contact is disabled in the CPU interaction dictionary.
        }
        if (satisfied) ++score.satisfied_ligand_areas;
        score.weighted_score += best;
    }
    for (std::uint32_t p = 0; p < protein.count; ++p)
        score.satisfied_protein_areas += coverage[p] != 0;
    score.valid = isfinite(score.weighted_score);
    output[index] = score;
}

} // namespace

void cuda_free_weighted_interaction_areas(GpuWeightedInteractionAreasDevice& device) {
    if (device.areas) cudaFree(device.areas);
    device = {};
}

void cuda_free_weighted_interaction_scores(GpuWeightedInteractionScoresDevice& device) {
    if (device.scores) cudaFree(device.scores);
    device = {};
}

bool cuda_upload_weighted_interaction_areas(const std::vector<GpuWeightedInteractionArea>& host, GpuWeightedInteractionAreasDevice& device) {
    if (host.size() > std::numeric_limits<std::uint32_t>::max() ||
        host.size() > std::numeric_limits<std::size_t>::max() / sizeof(GpuWeightedInteractionArea)) return false;
    GpuWeightedInteractionAreasDevice replacement;
    if (!host.empty()) {
        const std::size_t bytes = host.size() * sizeof(GpuWeightedInteractionArea);
        if (cudaMalloc(reinterpret_cast<void**>(&replacement.areas), bytes) != cudaSuccess)
            return false;
        if (cudaMemcpy(replacement.areas, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cuda_free_weighted_interaction_areas(replacement);
            return false;
        }
        replacement.count = static_cast<std::uint32_t>(host.size());
    }
    cuda_free_weighted_interaction_areas(device);
    device = replacement;
    return true;
}

bool cuda_score_weighted_interaction_device(
    const GpuWeightedInteractionAreasDevice& protein, const GpuWeightedInteractionAreasDevice& ligand,
    const GpuRigidTransformArrayDevice& transforms,
    const std::uint32_t* selected_pose_ids, std::uint32_t selected_count,
    const GpuWeightedInteractionWeights& weights, GpuWeightedInteractionScoresDevice& output) {
    cuda_free_weighted_interaction_scores(output);
    if ((protein.count && !protein.areas) || (ligand.count && !ligand.areas) ||
        (selected_count && !transforms.transforms) ||
        (!selected_pose_ids && selected_count > transforms.count)) return false;
    if (!selected_count) return true;
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (selected_count > max_size / sizeof(GpuWeightedInteractionScore) ||
        (protein.count && selected_count > max_size / protein.count)) return false;
    const std::size_t coverage_bytes = static_cast<std::size_t>(selected_count) * protein.count;
    unsigned char* coverage = nullptr;
    GpuWeightedInteractionScoresDevice replacement;
    if (coverage_bytes && cudaMalloc(reinterpret_cast<void**>(&coverage), coverage_bytes) != cudaSuccess)
        return false;
    if (cudaMalloc(reinterpret_cast<void**>(&replacement.scores),
                   static_cast<std::size_t>(selected_count) * sizeof(GpuWeightedInteractionScore)) != cudaSuccess) {
        if (coverage) cudaFree(coverage);
        return false;
    }
    constexpr std::uint32_t threads = 128;
    const auto blocks = static_cast<unsigned int>((static_cast<std::uint64_t>(selected_count) +
                                                  threads - 1) / threads);
    weighted_interaction_score_kernel<<<blocks, threads>>>(protein, ligand, transforms.transforms,
        transforms.count, selected_pose_ids, selected_count, weights, coverage, replacement.scores);
    const bool launched = cudaGetLastError() == cudaSuccess;
    const bool completed = launched && cudaDeviceSynchronize() == cudaSuccess;
    if (coverage) cudaFree(coverage);
    if (!completed) {
        cuda_free_weighted_interaction_scores(replacement);
        return false;
    }
    replacement.count = selected_count;
    output = replacement;
    return true;
}
