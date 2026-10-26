#include "gpu_docking_output.hpp"
#include "pose_score_ranking.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <Geometry/point.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace {

bool valid_pose(const GpuPipelinePose& pose) {
    return pose.transform.valid && pose.clash.valid && pose.fast_score.valid;
}

void require_finite_score(const GpuPipelinePose& pose) {
    if (!std::isfinite(pose.fast_score.weighted_score))
        throw std::invalid_argument("GPU docking pose has a nonfinite Fast score");
}

void validate_transform(const GpuRigidTransform& transform) {
    if (!transform.valid)
        throw std::invalid_argument("GPU docking pose has an invalid transform");
    for (double value : transform.rotation)
        if (!std::isfinite(value))
            throw std::invalid_argument("GPU docking rotation is nonfinite");
    for (double value : transform.translation)
        if (!std::isfinite(value))
            throw std::invalid_argument("GPU docking translation is nonfinite");
}

void validate_coordinates(const RDKit::ROMol& molecule) {
    if (!molecule.getNumConformers())
        throw std::invalid_argument("GPU docking molecule has no coordinate conformer");
    const auto& conformer = molecule.getConformer();
    for (unsigned int i = 0; i < molecule.getNumAtoms(); ++i) {
        const auto& point = conformer.getAtomPos(i);
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            throw std::invalid_argument("GPU docking molecule has nonfinite coordinates");
    }
}

void validate_reference(const RDKit::ROMol& ligand, const RDKit::ROMol& reference) {
    validate_coordinates(ligand);
    validate_coordinates(reference);
    // Compare atom indices directly; matching the graph is a check, not a remap.
    if (ligand.getNumAtoms() != reference.getNumAtoms() ||
        ligand.getNumBonds() != reference.getNumBonds())
        throw std::invalid_argument("RMSD reference has a different atom count or bond graph");
    std::size_t heavy_count = 0;
    for (unsigned int i = 0; i < ligand.getNumAtoms(); ++i) {
        const auto* atom = ligand.getAtomWithIdx(i);
        const auto* other = reference.getAtomWithIdx(i);
        if (atom->getAtomicNum() != other->getAtomicNum() ||
            atom->getIsotope() != other->getIsotope() ||
            atom->getFormalCharge() != other->getFormalCharge() ||
            atom->getIsAromatic() != other->getIsAromatic())
            throw std::invalid_argument("RMSD reference atom order, element, isotope or charge differs");
        if (atom->getAtomicNum() > 1) ++heavy_count;
    }
    for (const auto* bond : ligand.bonds()) {
        const auto* other = reference.getBondBetweenAtoms(
            bond->getBeginAtomIdx(), bond->getEndAtomIdx());
        if (!other || bond->getBondType() != other->getBondType() ||
            bond->getIsAromatic() != other->getIsAromatic())
            throw std::invalid_argument("RMSD reference bond connectivity or bond order differs");
    }
    if (!heavy_count)
        throw std::invalid_argument("Heavy-atom RMSD requires at least one heavy atom");
}

RDGeom::Point3D transformed_point(const RDGeom::Point3D& point,
                                const GpuRigidTransform& transform) {
    const auto& r = transform.rotation;
    const auto& t = transform.translation;
    RDGeom::Point3D result(
        r[0] * point.x + r[1] * point.y + r[2] * point.z + t[0],
        r[3] * point.x + r[4] * point.y + r[5] * point.z + t[1],
        r[6] * point.x + r[7] * point.y + r[8] * point.z + t[2]);
    if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z))
        throw std::invalid_argument("GPU docking transformed coordinates are nonfinite");
    return result;
}

double validated_rmsd(const RDKit::ROMol& ligand, const GpuRigidTransform& transform,
                      const RDKit::ROMol& reference) {
    const auto& original = ligand.getConformer();
    const auto& target = reference.getConformer();
    long double squared_sum = 0.0L;
    std::size_t count = 0;
    for (unsigned int i = 0; i < ligand.getNumAtoms(); ++i) {
        if (ligand.getAtomWithIdx(i)->getAtomicNum() <= 1) continue;
        const auto point = transformed_point(original.getAtomPos(i), transform);
        const auto& expected = target.getAtomPos(i);
        const long double dx = static_cast<long double>(point.x) - expected.x;
        const long double dy = static_cast<long double>(point.y) - expected.y;
        const long double dz = static_cast<long double>(point.z) - expected.z;
        squared_sum += dx * dx + dy * dy + dz * dz;
        ++count;
    }
    const double rmsd = static_cast<double>(std::sqrt(squared_sum / count));
    if (!std::isfinite(rmsd))
        throw std::invalid_argument("Heavy-atom RMSD is nonfinite");
    return rmsd;
}

std::string csv_string(const std::string& value) {
    std::string result = "\"";
    for (char character : value) {
        if (character == '"') result += '"';
        result += character;
    }
    return result + '"';
}

std::string numeric_property(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

void validate_scores(const GpuDockingPoseScores& scores) {
    const auto require_finite = [](double value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("GPU docking pose has a nonfinite rescoring value");
    };
    if (scores.weighted_interaction) require_finite(scores.weighted_interaction->weighted_score);
    if (scores.vinardo) {
        require_finite(scores.vinardo->total);
        require_finite(scores.vinardo->gaussian);
        require_finite(scores.vinardo->repulsion);
        require_finite(scores.vinardo->hydrophobic);
        require_finite(scores.vinardo->hydrogen_bond);
    }
    if (scores.vinardo_adjusted) require_finite(*scores.vinardo_adjusted);
    if (scores.hybrid) require_finite(*scores.hybrid);
}

const std::vector<std::string>& extended_properties() {
    static const std::vector<std::string> properties = {
        "pose_id", "weighted_interaction_score", "vinardo_like_score", "vinardo_gaussian",
        "vinardo_repulsion", "vinardo_hydrophobic", "vinardo_hydrogen_bond",
        "vinardo_adjusted_score", "hybrid_score"};
    return properties;
}

void write_extended_scores(std::ostream& rows, RDKit::RWMol& molecule,
                           const GpuDockingPoseScores& scores) {
    validate_scores(scores);
    for (const auto& property : extended_properties())
        if (molecule.hasProp(property)) molecule.clearProp(property);
    molecule.setProp("pose_id", std::to_string(scores.pose_id));
    rows << ',' << scores.pose_id;
    const auto write_score = [&](const char* property, std::optional<double> value) {
        rows << ',';
        if (value) {
            rows << *value;
            molecule.setProp(property, numeric_property(*value));
        }
    };
    write_score("weighted_interaction_score", scores.weighted_interaction
        ? std::optional<double>(scores.weighted_interaction->weighted_score) : std::nullopt);
    const auto& vinardo = scores.vinardo;
    write_score("vinardo_like_score", vinardo
        ? std::optional<double>(vinardo->total) : std::nullopt);
    write_score("vinardo_gaussian", vinardo
        ? std::optional<double>(vinardo->gaussian) : std::nullopt);
    write_score("vinardo_repulsion", vinardo
        ? std::optional<double>(vinardo->repulsion) : std::nullopt);
    write_score("vinardo_hydrophobic", vinardo
        ? std::optional<double>(vinardo->hydrophobic) : std::nullopt);
    write_score("vinardo_hydrogen_bond", vinardo
        ? std::optional<double>(vinardo->hydrogen_bond) : std::nullopt);
    write_score("vinardo_adjusted_score", scores.vinardo_adjusted);
    write_score("hybrid_score", scores.hybrid);
}

} // namespace

std::vector<GpuPipelinePose> select_top_gpu_poses(
    const std::vector<GpuPipelinePose>& poses, std::size_t k) {
    if (!k) throw std::invalid_argument("GPU docking top-k must be positive");
    std::vector<PoseScoreEntry> scores;
    std::vector<std::size_t> originals;
    scores.reserve(poses.size());
    originals.reserve(poses.size());
    for (std::size_t i = 0; i < poses.size(); ++i) {
        const auto& pose = poses[i];
        if (!valid_pose(pose)) continue;
        require_finite_score(pose);
        // The historical Fast tie-break is alignment ID, not survivor position.
        scores.push_back({pose.alignment_index, pose.fast_score.weighted_score});
        originals.push_back(i);
    }
    const auto ranked = top_pose_score_indices(scores, k);
    std::vector<GpuPipelinePose> result;
    result.reserve(ranked.size());
    for (std::size_t index : ranked) {
        // Keep the original GPU record, without converting coordinates through
        // the float-based interaction::point3D geometry helpers.
        result.push_back(poses[originals[index]]);
    }
    return result;
}

double gpu_pose_heavy_atom_rmsd(const RDKit::ROMol& ligand,
                              const GpuRigidTransform& transform,
                              const RDKit::ROMol& reference) {
    validate_transform(transform);
    validate_reference(ligand, reference);
    return validated_rmsd(ligand, transform, reference);
}

void validate_gpu_docking_reference(const RDKit::ROMol& ligand,
                                    const RDKit::ROMol& reference) {
    validate_reference(ligand, reference);
}

void write_gpu_docking_pose_header(std::ostream& csv, bool with_rmsd, bool extended_scores) {
    csv << "record_index,molecule_id,rank,alignment_index,fast_weighted_score,"
           "clash_fraction,occupied_atoms,r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz";
    if (with_rmsd) csv << ",rmsd_heavy_A";
    if (extended_scores)
        for (const auto& property : extended_properties()) csv << ',' << property;
    csv << '\n';
    if (!csv) throw std::runtime_error("Could not write GPU docking pose CSV header");
}

void write_gpu_docking_poses(std::ostream& csv, std::ostream& sdf,
                            const RDKit::RWMol& ligand,
                            const std::vector<GpuPipelinePose>& ranked,
                            std::size_t record_index, const std::string& molecule_id,
                            bool with_rmsd, const RDKit::ROMol* reference,
                            const std::vector<GpuDockingPoseScores>* scores) {
    if (!csv || !sdf) throw std::runtime_error("GPU docking output stream is not writable");
    if (reference && !with_rmsd)
        throw std::invalid_argument("RMSD reference supplied without the RMSD output column");
    if (scores && scores->size() != ranked.size())
        throw std::invalid_argument("GPU docking export scores do not match the ranked pose count");
    if (ranked.empty()) return;
    validate_coordinates(ligand);
    if (reference) validate_reference(ligand, *reference);
    // Stage a ligand's complete records before modifying the caller's streams.
    std::ostringstream rows, structures;
    rows << std::setprecision(17);
    RDKit::SDWriter writer(&structures, false);
    std::vector<std::string> properties = {
        "molecule_id", "record_index", "rank", "alignment_index",
        "fast_weighted_score", "clash_fraction", "occupied_atoms", "rmsd_heavy_A"};
    if (scores) properties.insert(properties.end(), extended_properties().begin(),
                                  extended_properties().end());
    writer.setProps(properties);
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        const auto& pose = ranked[i];
        if (!valid_pose(pose)) throw std::invalid_argument("Cannot export an invalid GPU docking pose");
        require_finite_score(pose);
        validate_transform(pose.transform);
        if (!std::isfinite(pose.clash.fraction))
            throw std::invalid_argument("GPU docking pose has a nonfinite clash fraction");
        RDKit::RWMol molecule(ligand);
        auto& conformer = molecule.getConformer();
        for (unsigned int atom = 0; atom < molecule.getNumAtoms(); ++atom)
            conformer.setAtomPos(atom, transformed_point(conformer.getAtomPos(atom), pose.transform));
        // SDF name occupies one line; retain the original ID as a property too.
        std::string name = molecule_id;
        std::replace(name.begin(), name.end(), '\n', ' ');
        std::replace(name.begin(), name.end(), '\r', ' ');
        molecule.setProp("_Name", name + " pose " + std::to_string(i + 1));
        molecule.setProp("molecule_id", molecule_id);
        molecule.setProp("record_index", std::to_string(record_index));
        molecule.setProp("rank", std::to_string(i + 1));
        molecule.setProp("alignment_index", std::to_string(pose.alignment_index));
        molecule.setProp("fast_weighted_score", numeric_property(pose.fast_score.weighted_score));
        molecule.setProp("clash_fraction", numeric_property(pose.clash.fraction));
        molecule.setProp("occupied_atoms", std::to_string(pose.clash.occupied_atoms));
        // Remove any same-named property inherited from an input SDF molecule.
        if (molecule.hasProp("rmsd_heavy_A")) molecule.clearProp("rmsd_heavy_A");
        rows << record_index << ',' << csv_string(molecule_id) << ',' << i + 1 << ','
             << pose.alignment_index << ',' << pose.fast_score.weighted_score << ','
             << pose.clash.fraction << ',' << pose.clash.occupied_atoms;
        for (double value : pose.transform.rotation) rows << ',' << value;
        for (double value : pose.transform.translation) rows << ',' << value;
        if (with_rmsd) {
            rows << ',';
            if (reference) {
                const double rmsd = validated_rmsd(ligand, pose.transform, *reference);
                rows << rmsd;
                molecule.setProp("rmsd_heavy_A", numeric_property(rmsd));
            }
        }
        if (scores) write_extended_scores(rows, molecule, scores->at(i));
        rows << '\n';
        writer.write(molecule);
    }
    writer.flush();
    if (!structures || !rows) throw std::runtime_error("Could not prepare GPU docking output");
    csv << rows.str();
    sdf << structures.str();
    if (!csv || !sdf) throw std::runtime_error("Could not write GPU docking pose output");
}
