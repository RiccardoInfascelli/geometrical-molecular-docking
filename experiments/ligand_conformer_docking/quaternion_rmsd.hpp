#ifndef CONFORMER_EXPERIMENT_QUATERNION_RMSD_HPP
#define CONFORMER_EXPERIMENT_QUATERNION_RMSD_HPP

#include <array>
#include <vector>

namespace conformer_experiment {

// Minimum RMSD after translation and a proper (determinant +1) rotation fitted
// by Horn's unit quaternion method. Every source[i] corresponds to target[i];
// callers select heavy atoms beforehand. No symmetry matching, permutations,
// reflection or SVD is used, and neither input is changed. A nonempty equal-size
// pair of finite coordinate arrays is required. One/two-point, collinear and
// coincident clouds are valid even when the optimal rotation is not unique.
// Invalid inputs throw invalid_argument; numerical failures throw runtime_error.
double quaternion_aligned_rmsd(
    const std::vector<std::array<double, 3>>& source,
    const std::vector<std::array<double, 3>>& target);

} // namespace conformer_experiment

#endif
