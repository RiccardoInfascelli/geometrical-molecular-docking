#include "cuda_backend.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


namespace
{

using Triangle = std::array<GpuPoint3, 3>;

struct AlignmentCase
{
    const char* name;
    Triangle source;
    Triangle target;
    bool exact_fit;
};


Eigen::Matrix3d point_matrix(const Triangle& points)
{
    Eigen::Matrix3d matrix;
    for (int i = 0; i < 3; ++i)
        matrix.col(i) << points[i].x, points[i].y, points[i].z;
    return matrix;
}


Triangle transformed_triangle(
    const Triangle& source,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation
)
{
    Triangle target{};
    for (int i = 0; i < 3; ++i) {
        const Eigen::Vector3d point = rotation *
            Eigen::Vector3d(source[i].x, source[i].y, source[i].z) + translation;
        target[i] = {point.x(), point.y(), point.z()};
    }
    return target;
}


double reference_residual(const AlignmentCase& test)
{
    Eigen::Matrix3d source = point_matrix(test.source);
    Eigen::Matrix3d target = point_matrix(test.target);
    source.colwise() -= source.rowwise().mean().eval();
    target.colwise() -= target.rowwise().mean().eval();

    const Eigen::Matrix3d covariance = source * target.transpose();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV
    );
    Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
    if ((svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0)
        correction(2, 2) = -1.0;
    const Eigen::Matrix3d rotation =
        svd.matrixV() * correction * svd.matrixU().transpose();
    return (rotation * source - target).squaredNorm();
}


bool check_transform(
    const AlignmentCase& test,
    const GpuRigidTransform& transform
)
{
    constexpr double point_tolerance = 1e-8;
    constexpr double rotation_tolerance = 1e-9;
    constexpr double residual_tolerance = 1e-9;

    Eigen::Matrix3d rotation;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            rotation(row, column) = transform.rotation[3 * row + column];
    const Eigen::Vector3d translation(
        transform.translation[0], transform.translation[1], transform.translation[2]
    );

    if (!transform.valid || !rotation.allFinite() || !translation.allFinite() ||
        std::abs(rotation.determinant() - 1.0) > rotation_tolerance ||
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() >
            rotation_tolerance) {
        std::cerr << test.name << ": invalid proper rotation\n";
        return false;
    }

    Eigen::Matrix3d fitted = rotation * point_matrix(test.source);
    fitted.colwise() += translation;
    const Eigen::Matrix3d error = fitted - point_matrix(test.target);
    const double residual = error.squaredNorm();
    const double optimum = reference_residual(test);

    // Degenerate triangles need not have a unique rotation. Compare the
    // least-squares objective instead of requiring identical matrix entries.
    if (!std::isfinite(residual) ||
        std::abs(residual - optimum) > residual_tolerance * (1.0 + optimum)) {
        std::cerr << test.name << ": residual " << residual
                  << " differs from the SVD reference " << optimum << '\n';
        return false;
    }
    if (test.exact_fit && error.cwiseAbs().maxCoeff() > point_tolerance) {
        std::cerr << test.name << ": transformed points do not match\n";
        return false;
    }
    return true;
}

} // namespace


int main()
{
    const Triangle source{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}};
    const Triangle scalene{{{-1.2, 0.7, 2.3}, {2.1, -0.4, 1.6}, {0.3, 2.8, -0.7}}};
    const Triangle near_collinear{{{0.0, 0.0, 0.0}, {1.0, 1e-5, 0.0}, {2.0, 0.0, 0.0}}};
    const Triangle collinear{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}};
    const Triangle coincident{{{-1.0, 2.0, 3.0}, {-1.0, 2.0, 3.0}, {-1.0, 2.0, 3.0}}};
    const Eigen::Matrix3d oblique_rotation = Eigen::AngleAxisd(
        0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized()
    ).toRotationMatrix();
    const Eigen::Vector3d translation(2.0, -3.0, 4.0);
    Triangle perturbed = transformed_triangle(scalene, oblique_rotation, translation);
    perturbed[0].x += 0.13;
    perturbed[1].y -= 0.08;
    perturbed[2].z += 0.04;

    const std::vector<AlignmentCase> cases{
        {"Translation", source,
            {{{2.0, -3.0, 4.0}, {3.0, -3.0, 4.0}, {2.0, -2.0, 4.0}}}, true},
        {"Z rotation", source,
            {{{5.0, 6.0, 7.0}, {5.0, 7.0, 7.0}, {4.0, 6.0, 7.0}}}, true},
        {"Oblique rotation", scalene,
            transformed_triangle(scalene, oblique_rotation, translation), true},
        {"Noncongruent triangles", scalene, perturbed, false},
        {"Near-collinear points", near_collinear,
            transformed_triangle(near_collinear, oblique_rotation, translation), false},
        {"Collinear points", collinear,
            transformed_triangle(collinear, oblique_rotation, translation), true},
        {"Coincident points", coincident,
            transformed_triangle(coincident, oblique_rotation, translation), true}
    };

    std::vector<GpuPoint3> ligand_points;
    std::vector<GpuPoint3> protein_points;
    std::vector<GpuAlignmentCandidate> alignments;
    for (const auto& test : cases) {
        const auto offset = static_cast<std::uint32_t>(ligand_points.size());
        ligand_points.insert(ligand_points.end(), test.source.begin(), test.source.end());
        protein_points.insert(protein_points.end(), test.target.begin(), test.target.end());
        alignments.push_back({{offset, offset + 1, offset + 2},
                              {offset, offset + 1, offset + 2}});
    }

    GpuPointArrayDevice device_ligand_points{};
    GpuPointArrayDevice device_protein_points{};
    const auto cleanup = [&]() {
        cuda_free_points(device_protein_points);
        cuda_free_points(device_ligand_points);
    };
    if (!cuda_upload_points(ligand_points, device_ligand_points) ||
        !cuda_upload_points(protein_points, device_protein_points)) {
        std::cerr << "Point upload failed\n";
        cleanup();
        return 1;
    }

    // Set DOCKING_KABSCH_METHOD before launch to run the same fixtures with
    // either backend. The host SVD reference is independent of that selector.
    std::vector<GpuRigidTransform> transforms;
    if (!cuda_compute_kabsch_transforms(device_ligand_points, device_protein_points,
                                        alignments, transforms) ||
        transforms.size() != cases.size()) {
        std::cerr << "CUDA Kabsch failed or returned the wrong count\n";
        cleanup();
        return 1;
    }
    for (std::size_t i = 0; i < cases.size(); ++i) {
        if (!check_transform(cases[i], transforms[i])) {
            cleanup();
            return 1;
        }
    }

    // An empty request clears prior host results; it is not a solver failure.
    if (!cuda_compute_kabsch_transforms(device_ligand_points, device_protein_points,
                                        {}, transforms) || !transforms.empty()) {
        std::cerr << "Empty alignment request failed\n";
        cleanup();
        return 1;
    }
    auto invalid_alignment = alignments.front();
    invalid_alignment.ligand_indices[0] = device_ligand_points.count;
    if (cuda_compute_kabsch_transforms(device_ligand_points, device_protein_points,
                                       {invalid_alignment}, transforms)) {
        std::cerr << "Out-of-range host alignment was accepted\n";
        cleanup();
        return 1;
    }

    cleanup();
    const char* selected_method = std::getenv("DOCKING_KABSCH_METHOD");
    std::cout << "CUDA Kabsch tests passed (method="
              << (selected_method ? selected_method : "default SVD") << ")\n";
    return 0;
}
