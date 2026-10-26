#include "cuda_pose_ranking.hpp"

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct InvalidFast {
    const GpuFastScore* fast;
    __host__ __device__ bool operator()(std::uint32_t i) const {
        return !fast[i].valid || !isfinite(fast[i].weighted_score);
    }
};
struct InvalidPoolPose {
    const GpuRigidTransform* transforms;
    const GpuClashScore* clash;
    const GpuFastScore* fast;
    __host__ __device__ bool operator()(std::uint32_t i) const {
        return !transforms[i].valid || !clash[i].valid || !fast[i].valid || !isfinite(fast[i].weighted_score);
    }
};
struct FastBefore {
    const GpuFastScore* fast;
    __host__ __device__ bool operator()(std::uint32_t a, std::uint32_t b) const {
        if (fast[a].weighted_score != fast[b].weighted_score)
            return fast[a].weighted_score > fast[b].weighted_score;
        return a < b;
    }
};
struct InvalidRescore {
    const GpuWeightedInteractionScore* weighted_interaction;
    const GpuVinardoScore* vinardo;
    const std::uint32_t* ids;
    std::uint32_t pool_count;
    int method;
    double alpha;
    __host__ __device__ bool operator()(std::uint32_t i) const {
        if (ids[i] >= pool_count) return true;
        if (method != 2 && (!weighted_interaction[i].valid || !isfinite(weighted_interaction[i].weighted_score))) return true;
        if (method != 1) {
            const auto& v = vinardo[i];
            if (!v.valid || !isfinite(v.total) || !isfinite(v.gaussian) ||
                !isfinite(v.repulsion) || !isfinite(v.hydrophobic) || !isfinite(v.hydrogen_bond)) return true;
            if (!isfinite(v.total + (alpha - 1.0) * v.repulsion)) return true;
        }
        return false;
    }
};
__global__ void prepare_values(const GpuWeightedInteractionScore* weighted_interaction, const GpuVinardoScore* vinardo,
    int method, double alpha, std::uint32_t count, double* e, double* reward) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    e[i] = method != 2 ? weighted_interaction[i].weighted_score : 0.0;
    reward[i] = method != 1 ? -(vinardo[i].total + (alpha - 1.0) * vinardo[i].repulsion) : 0.0;
}
struct AbsValue {
    __host__ __device__ double operator()(double value) const { return fabs(value); }
};
struct Moment { std::uint64_t count{}; double mean{}, m2{}; };
struct ScaledMoment {
    double scale;
    __host__ __device__ Moment operator()(double x) const {
        return {1, scale == 0.0 ? 0.0 : x / scale, 0.0};
    }
};
struct MergeMoment {
    __host__ __device__ Moment operator()(const Moment& a, const Moment& b) const {
        if (!a.count) return b;
        if (!b.count) return a;
        const double n = static_cast<double>(a.count + b.count);
        const double delta = b.mean - a.mean;
        return {a.count + b.count, a.mean + delta * (static_cast<double>(b.count) / n),
            a.m2 + b.m2 + delta * delta * (static_cast<double>(a.count) / n) * b.count};
    }
};
__device__ double zscore(double value, double scale, Moment moment) {
    if (scale == 0.0 || moment.m2 == 0.0) return 0.0;
    return (value / scale - moment.mean) / sqrt(moment.m2 / moment.count);
}
__global__ void hybrid_values(const double* weighted_interaction, const double* reward, std::uint32_t count,
    double e_scale, double v_scale, Moment e, Moment v, double lambda, double* hybrid) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        hybrid[i] = lambda * zscore(weighted_interaction[i], e_scale, e) + (1.0 - lambda) * zscore(reward[i], v_scale, v);
}
struct Nonfinite {
    __host__ __device__ bool operator()(double value) const { return !isfinite(value); }
};
struct ScoreBefore {
    const double* score;
    const std::uint32_t* pose_ids;
    __host__ __device__ bool operator()(std::uint32_t a, std::uint32_t b) const {
        if (score[a] != score[b]) return score[a] > score[b];
        if (pose_ids[a] != pose_ids[b]) return pose_ids[a] < pose_ids[b];
        return a < b;
    }
};
__global__ void gather_top(const std::uint32_t* ranked, const std::uint32_t* pose_ids,
    const GpuRigidTransform* transforms, const GpuClashScore* clash,
    const std::uint32_t* alignment_ids, const GpuFastScore* fast,
    const GpuWeightedInteractionScore* weighted_interaction, const GpuVinardoScore* vinardo,
    const double* reward, const double* hybrid, int method,
    std::uint32_t count, GpuRankedRescoredPose* output) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto local = ranked[i], id = pose_ids[local];
    GpuRankedRescoredPose row{};
    row.pose_id = id;
    row.pose = {alignment_ids[id], transforms[id], clash[id], fast[id]};
    if (method != 2) row.weighted_interaction = weighted_interaction[local];
    if (method != 1) { row.vinardo = vinardo[local]; row.adjusted = -reward[local]; }
    if (method == 3) row.hybrid = hybrid[local];
    output[i] = row;
}
} // namespace

void cuda_free_rescoring_selection(GpuRescoringSelection& selection) {
    if (selection.pose_ids) cudaFree(selection.pose_ids);
    selection = {};
}

bool cuda_select_rescoring_pool(const GpuFastScoreArrayDevice& fast,
    std::uint32_t top_n, GpuRescoringSelection& selection) {
    cuda_free_rescoring_selection(selection);
    if (!fast.count) return true;
    if (!fast.scores) return false;
    try {
        const auto first = thrust::make_counting_iterator<std::uint32_t>(0);
        if (thrust::count_if(thrust::device, first, first + fast.count, InvalidFast{fast.scores})) return false;
        if (cudaMalloc(reinterpret_cast<void**>(&selection.pose_ids), sizeof(std::uint32_t) * fast.count) != cudaSuccess) return false;
        auto ids = thrust::device_pointer_cast(selection.pose_ids);
        thrust::sequence(ids, ids + fast.count);
        selection.count = top_n ? std::min(top_n, fast.count) : fast.count;
        if (selection.count < fast.count) {
            thrust::sort(ids, ids + fast.count, FastBefore{fast.scores});
            thrust::sort(ids, ids + selection.count);
        }
        if (cudaDeviceSynchronize() == cudaSuccess) return true;
    } catch (...) {}
    cuda_free_rescoring_selection(selection);
    return false;
}

bool cuda_rank_rescored_poses(const GpuPosePipelineDeviceResult& pool,
    const GpuRescoringSelection& selection, const GpuWeightedInteractionScoresDevice& weighted_interaction,
    const GpuVinardoScoresDevice& vinardo, int method, double alpha, double lambda,
    std::uint32_t top_k, std::vector<GpuRankedRescoredPose>& output, GpuRankingTimings* timings) {
    output.clear();
    if (timings) *timings = {};
    if (!top_k || method < 1 || method > 3 || !std::isfinite(alpha) || alpha < 0.0 ||
        !std::isfinite(lambda) || lambda < 0.0 || lambda > 1.0) return false;
    const auto n = selection.count;
    if (!n) return true;
    if (!selection.pose_ids || pool.survivors.count != pool.fast_scores.count ||
        !pool.survivors.transforms || !pool.survivors.clash_scores ||
        !pool.survivors.alignment_indices || !pool.fast_scores.scores ||
        (method != 2 && (!weighted_interaction.scores || weighted_interaction.count != n)) ||
        (method != 1 && (!vinardo.scores || vinardo.count != n))) return false;
    try {
        auto start = Clock::now();
        const auto first = thrust::make_counting_iterator<std::uint32_t>(0);
        if (thrust::count_if(thrust::device, first, first + pool.survivors.count,
            InvalidPoolPose{pool.survivors.transforms, pool.survivors.clash_scores, pool.fast_scores.scores})) return false;
        if (thrust::count_if(thrust::device, first, first + n, InvalidRescore{
            weighted_interaction.scores, vinardo.scores, selection.pose_ids, pool.survivors.count, method, alpha})) return false;
        thrust::device_vector<double> e(n), reward(n), hybrid(method == 3 ? n : 0);
        prepare_values<<<(n - 1) / 128 + 1, 128>>>(weighted_interaction.scores, vinardo.scores, method, alpha,
            n, thrust::raw_pointer_cast(e.data()), thrust::raw_pointer_cast(reward.data()));
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
        if (timings) timings->score_preparation_seconds = seconds(start);
        start = Clock::now();
        if (method == 3) {
            // Normalize once over all selected poses, before top-k. Only the
            // reduced scales and moments cross the host; pose values stay here.
            const double es = thrust::transform_reduce(e.begin(), e.end(), AbsValue{}, 0.0, thrust::maximum<double>{});
            const double vs = thrust::transform_reduce(reward.begin(), reward.end(), AbsValue{}, 0.0, thrust::maximum<double>{});
            const auto em = thrust::transform_reduce(e.begin(), e.end(), ScaledMoment{es}, Moment{}, MergeMoment{});
            const auto vm = thrust::transform_reduce(reward.begin(), reward.end(), ScaledMoment{vs}, Moment{}, MergeMoment{});
            if (!std::isfinite(em.mean) || !std::isfinite(vm.mean) || !std::isfinite(em.m2) ||
                !std::isfinite(vm.m2) || em.m2 < 0.0 || vm.m2 < 0.0) return false;
            hybrid_values<<<(n - 1) / 128 + 1, 128>>>(thrust::raw_pointer_cast(e.data()),
                thrust::raw_pointer_cast(reward.data()), n, es, vs, em, vm, lambda,
                thrust::raw_pointer_cast(hybrid.data()));
            if (cudaGetLastError() != cudaSuccess ||
                thrust::count_if(hybrid.begin(), hybrid.end(), Nonfinite{})) return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        if (timings && method == 3) timings->hybrid_seconds = seconds(start);

        start = Clock::now();
        thrust::device_vector<std::uint32_t> indices(n);
        thrust::sequence(indices.begin(), indices.end());
        const double* keys = method == 1 || (method == 3 && lambda == 1.0)
            ? thrust::raw_pointer_cast(e.data())
            : method == 2 || lambda == 0.0 ? thrust::raw_pointer_cast(reward.data())
            : thrust::raw_pointer_cast(hybrid.data());
        thrust::sort(indices.begin(), indices.end(), ScoreBefore{keys, selection.pose_ids});
        const auto count = std::min(top_k, n);
        thrust::device_vector<GpuRankedRescoredPose> selected(count);
        gather_top<<<(count - 1) / 128 + 1, 128>>>(thrust::raw_pointer_cast(indices.data()),
            selection.pose_ids, pool.survivors.transforms, pool.survivors.clash_scores,
            pool.survivors.alignment_indices, pool.fast_scores.scores, weighted_interaction.scores, vinardo.scores,
            thrust::raw_pointer_cast(reward.data()), method == 3 ? thrust::raw_pointer_cast(hybrid.data()) : nullptr,
            method, count, thrust::raw_pointer_cast(selected.data()));
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
        if (timings) timings->ranking_seconds = seconds(start);
        start = Clock::now();
        output.resize(count);
        if (cudaMemcpy(output.data(), thrust::raw_pointer_cast(selected.data()),
                       sizeof(GpuRankedRescoredPose) * count, cudaMemcpyDeviceToHost) != cudaSuccess) {
            output.clear(); return false;
        }
        if (timings) timings->download_seconds = seconds(start);
        return true;
    } catch (...) { output.clear(); return false; }
}
