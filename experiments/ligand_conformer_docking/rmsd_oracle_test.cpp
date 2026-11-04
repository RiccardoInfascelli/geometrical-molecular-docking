#include "ensemble_ranking.hpp"
#include "gpu_docking_output.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/RWMol.h>
#include <Geometry/point.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= 1e-12,
            "Unexpected direct heavy-atom RMSD");
}

void rejects(const std::function<void()>& operation) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument for an invalid oracle reference or mapping");
}

RDKit::RWMol molecule() {
    RDKit::RWMol result;
    for (const auto atomic_number : {6u, 8u, 8u, 1u})
        result.addAtom(new RDKit::Atom(atomic_number), true, true);
    for (const auto index : {1u, 2u, 3u})
        result.addBond(0, index, RDKit::Bond::SINGLE);
    auto* coordinates = new RDKit::Conformer(4);
    coordinates->set3D(true);
    coordinates->setAtomPos(0, {0, 0, 0});
    coordinates->setAtomPos(1, {1, 0, 0});
    coordinates->setAtomPos(2, {-1, 0, 0});
    coordinates->setAtomPos(3, {0, 0, 1});
    result.addConformer(coordinates, true);
    result.updatePropertyCache(false);
    return result;
}

GpuRigidTransform identity() {
    GpuRigidTransform transform{};
    transform.valid = 1;
    transform.rotation[0] = transform.rotation[4] = transform.rotation[8] = 1.0;
    return transform;
}

void test_direct_heavy_rmsd() {
    const auto reference = molecule();
    auto generated = molecule();
    const auto transform = identity();
    validate_gpu_docking_reference(generated, reference);
    close(gpu_pose_heavy_atom_rmsd(generated, transform, reference), 0.0);

    generated.getConformer().setAtomPos(3, {100, -20, 90});
    close(gpu_pose_heavy_atom_rmsd(generated, transform, reference), 0.0);
    auto displaced = transform;
    displaced.translation[0] = 3.0;
    close(gpu_pose_heavy_atom_rmsd(generated, displaced, reference), 3.0);

    // These oxygen atoms are graph-equivalent. Fixed-index evaluation retains
    // their mismatch instead of swapping atoms or fitting another rotation.
    generated.getConformer().setAtomPos(1, {-1, 0, 0});
    generated.getConformer().setAtomPos(2, {1, 0, 0});
    close(gpu_pose_heavy_atom_rmsd(generated, transform, reference), std::sqrt(8.0 / 3.0));
}

void test_reference_validation() {
    const auto reference = molecule();
    const auto generated = molecule();
    const auto transform = identity();
    RDKit::RWMol missing_reference;
    rejects([&] { gpu_pose_heavy_atom_rmsd(generated, transform, missing_reference); });
    auto missing_coordinates = reference;
    missing_coordinates.clearConformers();
    rejects([&] { gpu_pose_heavy_atom_rmsd(generated, transform, missing_coordinates); });
    auto wrong_element = reference;
    wrong_element.getAtomWithIdx(1)->setAtomicNum(7);
    rejects([&] { validate_gpu_docking_reference(generated, wrong_element); });
    auto wrong_graph = reference;
    wrong_graph.removeBond(0, 2);
    wrong_graph.addBond(1, 2, RDKit::Bond::SINGLE);
    rejects([&] { gpu_pose_heavy_atom_rmsd(generated, transform, wrong_graph); });
    auto wrong_count = reference;
    wrong_count.removeAtom(3);
    rejects([&] { gpu_pose_heavy_atom_rmsd(generated, transform, wrong_count); });
    auto nonfinite = reference;
    nonfinite.getConformer().setAtomPos(0, {std::numeric_limits<double>::quiet_NaN(), 0, 0});
    rejects([&] { gpu_pose_heavy_atom_rmsd(generated, transform, nonfinite); });
    auto invalid_transform = transform;
    invalid_transform.translation[1] = std::numeric_limits<double>::infinity();
    rejects([&] { gpu_pose_heavy_atom_rmsd(generated, invalid_transform, reference); });
}

void test_oracle_gap_and_baseline() {
    EnsemblePose selected;
    selected.conformer_id = 1;
    selected.pose_id = 0;
    selected.rmsd = 3.0;
    selected.scored.pose.fast_weighted_score = 100;
    EnsemblePose available = selected;
    available.conformer_id = 2;
    available.pose_id = 9;
    available.rmsd = 0.5;
    available.scored.pose.fast_weighted_score = -1;
    const std::vector<EnsemblePose> all{selected, available};
    const auto fast = rank_ensemble_poses(all, PoseScoreMethod::Fast);
    const auto oracle = rank_ensemble_oracle(all);
    require(fast.front().source_index == 0 && oracle.front().source_index == 1,
            "Oracle did not distinguish available accuracy from score-selected accuracy");
    close(all[fast.front().source_index].rmsd - oracle.front().score, 2.5);
    const auto local = rank_ensemble_oracle({selected});
    close(local.front().score, 3.0);
    EnsemblePose baseline = available;
    baseline.conformer_id = 0;
    baseline.rmsd = 0.0;
    rejects([&] { rank_ensemble_oracle({selected, available, baseline}); });
    close(rank_ensemble_oracle(all).front().score, 0.5);
    for (const auto invalid : {std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity(), -0.1}) {
        available.rmsd = invalid;
        rejects([&] { rank_ensemble_oracle({selected, available}); });
    }
}
} // namespace

int main() {
    try {
        test_direct_heavy_rmsd();
        test_reference_validation();
        test_oracle_gap_and_baseline();
        std::cout << "RMSD oracle, direct heavy-atom mapping and reference validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
