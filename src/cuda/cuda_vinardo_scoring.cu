#include "cuda_vinardo_scoring.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace {

__device__ double hydrophobic_ramp(double d) {
    if (d <= 0.0) return 1.0;
    if (d >= 2.5) return 0.0;
    return 1.0 - d / 2.5;
}

__device__ double hbond_ramp(double d) {
    if (d <= -0.6) return 1.0;
    if (d >= 0.0) return 0.0;
    return -d / 0.6;
}

__global__ void vinardo_score_kernel(
    const GpuVinardoAtom* receptor, std::uint32_t receptor_count,
    const GpuVinardoAtom* ligand, std::uint32_t ligand_count,
    const GpuRigidTransform* transforms, std::uint32_t transform_count,
    const std::uint32_t* selected_ids, std::uint32_t selected_count,
    GpuVinardoScore* output)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= selected_count) return;
    const std::uint32_t pose_id = selected_ids ? selected_ids[index] : index;
    GpuVinardoScore score{};
    if (pose_id >= transform_count || !transforms[pose_id].valid) {
        output[index] = score;
        return;
    }
    const auto transform = transforms[pose_id];
    for (std::uint32_t i = 0; i < ligand_count; ++i) {
        const auto atom = ligand[i];
        // CPU Vinardo goes RDKit double -> float point -> double arithmetic ->
        // float transformed point. Receptor coordinates remain RDKit doubles.
        const double x = static_cast<double>(static_cast<float>(atom.x));
        const double y = static_cast<double>(static_cast<float>(atom.y));
        const double z = static_cast<double>(static_cast<float>(atom.z));
        const double px = static_cast<double>(static_cast<float>(
            transform.rotation[0] * x + transform.rotation[1] * y +
            transform.rotation[2] * z + transform.translation[0]));
        const double py = static_cast<double>(static_cast<float>(
            transform.rotation[3] * x + transform.rotation[4] * y +
            transform.rotation[5] * z + transform.translation[1]));
        const double pz = static_cast<double>(static_cast<float>(
            transform.rotation[6] * x + transform.rotation[7] * y +
            transform.rotation[8] * z + transform.translation[2]));

        for (std::uint32_t j = 0; j < receptor_count; ++j) {
            const auto receptor_atom = receptor[j];
            const double dx = px - receptor_atom.x;
            const double dy = py - receptor_atom.y;
            const double dz = pz - receptor_atom.z;
            const double squared_distance = dx*dx + dy*dy + dz*dz;
            if (squared_distance >= 64.0) continue;
            const double d = sqrt(squared_distance) - (atom.radius + receptor_atom.radius);
            const double x_surface = d / 0.8;
            score.gaussian += -0.045 * exp(-(x_surface*x_surface));
            if (d < 0.0) score.repulsion += 0.8 * d*d;
            if (atom.hydrophobic && receptor_atom.hydrophobic)
                score.hydrophobic += -0.035 * hydrophobic_ramp(d);
            if ((atom.donor && receptor_atom.acceptor) ||
                (atom.acceptor && receptor_atom.donor))
                score.hydrogen_bond += -0.6 * hbond_ramp(d);
        }
    }
    score.total = score.gaussian + score.repulsion +
                  score.hydrophobic + score.hydrogen_bond;
    score.valid = isfinite(score.total) && isfinite(score.gaussian) &&
        isfinite(score.repulsion) && isfinite(score.hydrophobic) &&
        isfinite(score.hydrogen_bond);
    output[index] = score;
}

}

void cuda_free_vinardo_atoms(GpuVinardoAtomsDevice& device) {
    if (device.atoms) cudaFree(device.atoms);
    device = {};
}

void cuda_free_vinardo_scores(GpuVinardoScoresDevice& device) {
    if (device.scores) cudaFree(device.scores);
    device = {};
}

bool cuda_upload_vinardo_atoms(const std::vector<GpuVinardoAtom>& atoms,
                               GpuVinardoAtomsDevice& device)
{
    if (atoms.size() > std::numeric_limits<std::uint32_t>::max()) return false;
    if (atoms.empty()) {
        cuda_free_vinardo_atoms(device);
        return true;
    }
    GpuVinardoAtomsDevice replacement{};
    const auto bytes = atoms.size() * sizeof(GpuVinardoAtom);
    if (cudaMalloc(reinterpret_cast<void**>(&replacement.atoms), bytes) != cudaSuccess)
        return false;
    if (cudaMemcpy(replacement.atoms, atoms.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_free_vinardo_atoms(replacement);
        return false;
    }
    replacement.count = static_cast<std::uint32_t>(atoms.size());
    cuda_free_vinardo_atoms(device);
    device = replacement;
    return true;
}

bool cuda_score_vinardo_device(
    const GpuVinardoAtomsDevice& receptor,
    const GpuVinardoAtomsDevice& ligand,
    const GpuRigidTransformArrayDevice& transforms,
    const std::uint32_t* selected_pose_ids,
    std::uint32_t selected_count,
    GpuVinardoScoresDevice& output)
{
    cuda_free_vinardo_scores(output);
    if (selected_count == 0) return true;
    if (!transforms.transforms ||
        (receptor.count && !receptor.atoms) || (ligand.count && !ligand.atoms) ||
        selected_count > transforms.count ||
        (!selected_pose_ids && selected_count != transforms.count))
        return false;

    GpuVinardoScoresDevice result{};
    const auto bytes = static_cast<std::size_t>(selected_count) * sizeof(GpuVinardoScore);
    if (cudaMalloc(reinterpret_cast<void**>(&result.scores), bytes) != cudaSuccess)
        return false;
    constexpr std::uint32_t block_size = 128;
    const auto blocks = static_cast<std::uint32_t>(
        (static_cast<std::size_t>(selected_count) + block_size - 1) / block_size);
    vinardo_score_kernel<<<blocks, block_size>>>(
        receptor.atoms, receptor.count, ligand.atoms, ligand.count,
        transforms.transforms, transforms.count, selected_pose_ids, selected_count,
        result.scores);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cuda_free_vinardo_scores(result);
        return false;
    }
    result.count = selected_count;
    output = result;
    return true;
}
