#ifndef RIGID_ALIGNMENT_HPP
#define RIGID_ALIGNMENT_HPP

#include "geometry.hpp"

#include <array>

// Fits three ordered point pairs in double. SVD is the default; setting
// DOCKING_KABSCH_METHOD=quaternion selects the historical Jacobi quaternion fit.
// The environment is read once on the first CPU call; other values select SVD.
// Neither method validates rank or finite coordinates. SVD returns true after
// fitting; quaternion retains its norm check, without a convergence status.
bool rigid_transform_from_points(
    const std::array<interaction::point3D, 3>& source,
    const std::array<interaction::point3D, 3>& target,
    RigidTransform& transform
);

#endif // RIGID_ALIGNMENT_HPP
