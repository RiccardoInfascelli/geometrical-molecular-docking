#include "geometry.hpp"

#include <interaction/geometric_features.hpp>

#include <GraphMol/Atom.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/ROMol.h>

#include <cmath>
#include <stdexcept>

std::array<double, 3> heavy_atom_centroid(const RDKit::ROMol& molecule)
{
    if (!molecule.getNumConformers())
        throw std::invalid_argument("Pose center distance requires molecular coordinates");
    const auto& conformer = molecule.getConformer();
    std::array<long double, 3> sum{};
    std::size_t count = 0;
    for (const auto* atom : molecule.atoms()) {
        if (atom->getAtomicNum() <= 1) continue;
        const auto& point = conformer.getAtomPos(atom->getIdx());
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            throw std::invalid_argument("Pose center distance requires finite heavy-atom coordinates");
        sum[0] += point.x;
        sum[1] += point.y;
        sum[2] += point.z;
        ++count;
    }
    if (!count)
        throw std::invalid_argument("Pose center distance requires at least one heavy atom");
    std::array<double, 3> center{};
    for (std::size_t i = 0; i < center.size(); ++i) {
        center[i] = static_cast<double>(sum[i] / count);
        if (!std::isfinite(center[i]))
            throw std::invalid_argument("Pose center distance has a nonfinite centroid");
    }
    return center;
}

double pose_center_distance(const std::array<double, 3>& ligand_center,
    const RigidTransform& transform, const std::array<double, 3>& reference_center)
{
    for (const auto& center : {ligand_center, reference_center})
        for (double value : center)
            if (!std::isfinite(value))
                throw std::invalid_argument("Pose center distance has a nonfinite centroid");
    for (double value : transform.rotation)
        if (!std::isfinite(value))
            throw std::invalid_argument("Pose center distance has a nonfinite rotation");
    std::array<double, 3> displacement{};
    for (std::size_t i = 0; i < displacement.size(); ++i) {
        if (!std::isfinite(transform.translation[i]))
            throw std::invalid_argument("Pose center distance has a nonfinite translation");
        displacement[i] = transform.rotation[3 * i] * ligand_center[0] +
            transform.rotation[3 * i + 1] * ligand_center[1] +
            transform.rotation[3 * i + 2] * ligand_center[2] + transform.translation[i] - reference_center[i];
    }
    const double distance = std::hypot(displacement[0], displacement[1], displacement[2]);
    if (!std::isfinite(distance))
        throw std::invalid_argument("Pose center distance is nonfinite");
    return distance;
}

double compute_ligand_diameter(
    const std::vector<interaction::interaction_area>& ligand_areas)
{
    double max_distance = 0.0;

    for (std::size_t i = 0; i < ligand_areas.size(); ++i) {

        for (std::size_t j = i + 1;
             j < ligand_areas.size();
             ++j) {

            const double distance =
                std::sqrt(
                    interaction::distance2(
                        ligand_areas[i].centroid,
                        ligand_areas[j].centroid
                    )
                );

            if (distance > max_distance)
                max_distance = distance;
        }
    }

    return max_distance;
}

interaction::point3D transform_point(
    const interaction::point3D& point,
    const RigidTransform& transform)
{
    const double x =
        static_cast<double>(point.x);

    const double y =
        static_cast<double>(point.y);

    const double z =
        static_cast<double>(point.z);

    return interaction::point3D{
        static_cast<interaction::coordinate_type>(
            transform.rotation[0] * x +
            transform.rotation[1] * y +
            transform.rotation[2] * z +
            transform.translation[0]
        ),

        static_cast<interaction::coordinate_type>(
            transform.rotation[3] * x +
            transform.rotation[4] * y +
            transform.rotation[5] * z +
            transform.translation[1]
        ),

        static_cast<interaction::coordinate_type>(
            transform.rotation[6] * x +
            transform.rotation[7] * y +
            transform.rotation[8] * z +
            transform.translation[2]
        )
    };
}


std::vector<MatchingArea> transform_matching_areas(
    const std::vector<MatchingArea>& areas,
    const RigidTransform& transform)
{
    std::vector<MatchingArea> transformed =
        areas;

    for (auto& area : transformed) {

        area.centroid =
            transform_point(
                area.centroid,
                transform
            );
    }

    return transformed;
}

std::vector<interaction::interaction_area>
transform_interaction_areas(
    const std::vector<interaction::interaction_area>& areas,
    const RigidTransform& transform)
{
    auto transformed = areas;

    for (auto& area : transformed) {

        area.centroid =
            transform_point(
                area.centroid,
                transform
            );

        // WeightedInteraction scoring also needs the transformed atoms that define orientation.
        for (std::size_t i = 0;
             i < area.num_atoms;
             ++i) {

            const interaction::point3D point{
                area.x[i],
                area.y[i],
                area.z[i]
            };

            const auto transformed_point =
                transform_point(
                    point,
                    transform
                );

            area.x[i] =
                transformed_point.x;

            area.y[i] =
                transformed_point.y;

            area.z[i] =
                transformed_point.z;
        }
    }

    return transformed;
}
