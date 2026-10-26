#include "gpu_docking_output.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/RWMol.h>
#include <Geometry/point.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double tolerance = 1e-12) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error("Numeric mismatch: " + std::to_string(actual) +
                                 " versus " + std::to_string(expected));
}

void rejects(const std::function<void()>& operation) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument");
}

RDKit::RWMol molecule() {
    RDKit::RWMol result;
    result.addAtom(new RDKit::Atom(6), true, true);
    result.addAtom(new RDKit::Atom(8), true, true);
    result.addAtom(new RDKit::Atom(1), true, true);
    result.addBond(0, 1, RDKit::Bond::SINGLE);
    result.addBond(0, 2, RDKit::Bond::SINGLE);
    auto* conformer = new RDKit::Conformer(3);
    conformer->set3D(true);
    conformer->setAtomPos(0, RDGeom::Point3D(1, 0, 0));
    conformer->setAtomPos(1, RDGeom::Point3D(0, 2, 0));
    conformer->setAtomPos(2, RDGeom::Point3D(9, 9, 9));
    result.addConformer(conformer, true);
    result.updatePropertyCache(false);
    return result;
}

GpuPipelinePose pose(std::uint32_t index, double score) {
    GpuPipelinePose result;
    result.alignment_index = index;
    result.transform.valid = result.clash.valid = result.fast_score.valid = 1;
    result.transform.rotation[0] = result.transform.rotation[4] = result.transform.rotation[8] = 1;
    result.fast_score.weighted_score = score;
    result.clash.occupied_atoms = 1;
    result.clash.total_atoms = 3;
    result.clash.fraction = 1.0 / 3.0;
    return result;
}

std::vector<std::string> csv_fields(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"'; ++i;
            } else quoted = !quoted;
        } else if (line[i] == ',' && !quoted) {
            fields.push_back(field); field.clear();
        } else field += line[i];
    }
    require(!quoted, "Unterminated CSV string");
    fields.push_back(field);
    return fields;
}

void test_selection() {
    const std::vector<GpuPipelinePose> input = {pose(3, 5), pose(4, 7), pose(2, 7), pose(1, -2)};
    const auto ranked = select_top_gpu_poses(input, 2);
    require(ranked.size() == 2, "Top-k count differs");
    require(ranked[0].alignment_index == 2 && ranked[1].alignment_index == 4,
            "Fast ranking or alignment tie break differs");
    require(input[0].alignment_index == 3, "Selection mutated its input");
    require(select_top_gpu_poses(input, 100).size() == input.size(), "Large k discarded poses");
    require(select_top_gpu_poses({}, 1).empty(), "Empty input returned a pose");
    rejects([&] { select_top_gpu_poses(input, 0); });
    rejects([] { select_top_gpu_poses({}, 0); });
    for (int field = 0; field < 3; ++field) {
        auto invalid = pose(0, 1000);
        if (field == 0) invalid.transform.valid = 0;
        if (field == 1) invalid.clash.valid = 0;
        if (field == 2) invalid.fast_score.valid = 0;
        auto with_invalid = input;
        with_invalid.push_back(invalid);
        require(select_top_gpu_poses(with_invalid, 100).size() == input.size(),
                "Invalid pose entered ranking");
    }
    for (double score : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
        auto nonfinite = input;
        nonfinite.push_back(pose(9, score));
        rejects([&] { select_top_gpu_poses(nonfinite, 1); });
    }
    auto precise = pose(6, 100);
    precise.transform.translation[0] = 0.12345678901234567;
    precise.transform.rotation[1] = 0.012345678901234567;
    const auto original = select_top_gpu_poses({precise, pose(6, 1)}, 2);
    require(original.size() == 2 &&
            original[0].transform.translation[0] == precise.transform.translation[0] &&
            original[0].transform.rotation[1] == precise.transform.rotation[1] &&
            original[1].fast_score.weighted_score == 1,
            "Shared ranking changed original GPU transform precision or duplicate-ID mapping");
}

void test_rmsd() {
    const auto ligand = molecule();
    const auto identity = pose(0, 1).transform;
    validate_gpu_docking_reference(ligand, ligand);
    close(gpu_pose_heavy_atom_rmsd(ligand, identity, ligand), 0);
    auto translated = identity;
    translated.translation[0] = 1;
    translated.translation[1] = 2;
    translated.translation[2] = 2;
    close(gpu_pose_heavy_atom_rmsd(ligand, translated, ligand), 3);
    auto rotated = identity;
    rotated.rotation[0] = rotated.rotation[4] = 0;
    rotated.rotation[1] = -1;
    rotated.rotation[3] = 1;
    close(gpu_pose_heavy_atom_rmsd(ligand, rotated, ligand), std::sqrt(5.0));
    auto moved_hydrogen = ligand;
    moved_hydrogen.getConformer().setAtomPos(2, RDGeom::Point3D(-100, 50, 900));
    close(gpu_pose_heavy_atom_rmsd(ligand, identity, moved_hydrogen), 0);
    auto matching_reference = ligand;
    matching_reference.getConformer().setAtomPos(0, RDGeom::Point3D(2, 2, 2));
    matching_reference.getConformer().setAtomPos(1, RDGeom::Point3D(1, 4, 2));
    close(gpu_pose_heavy_atom_rmsd(ligand, translated, matching_reference), 0);
    for (int mismatch = 0; mismatch < 5; ++mismatch) {
        auto reference = ligand;
        if (mismatch == 0) reference.getAtomWithIdx(0)->setAtomicNum(8);
        if (mismatch == 1) reference.getAtomWithIdx(0)->setIsotope(13);
        if (mismatch == 2) reference.getAtomWithIdx(0)->setFormalCharge(1);
        if (mismatch == 3) reference.getBondWithIdx(0)->setBondType(RDKit::Bond::DOUBLE);
        if (mismatch == 4) {
            reference.removeBond(0, 2);
            reference.addBond(1, 2, RDKit::Bond::SINGLE);
        }
        rejects([&] { validate_gpu_docking_reference(ligand, reference); });
        rejects([&] { gpu_pose_heavy_atom_rmsd(ligand, identity, reference); });
    }
    auto no_conformer = ligand;
    no_conformer.clearConformers();
    rejects([&] { gpu_pose_heavy_atom_rmsd(ligand, identity, no_conformer); });
    auto invalid = identity;
    invalid.valid = 0;
    rejects([&] { gpu_pose_heavy_atom_rmsd(ligand, invalid, ligand); });
    auto nonfinite = identity;
    nonfinite.rotation[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { gpu_pose_heavy_atom_rmsd(ligand, nonfinite, ligand); });
}

void test_export() {
    const auto ligand = molecule();
    auto exported = pose(12, 4.25);
    exported.transform.rotation[0] = exported.transform.rotation[4] = 0;
    exported.transform.rotation[1] = -1;
    exported.transform.rotation[3] = 1;
    exported.transform.translation[0] = 1;
    exported.transform.translation[1] = 2;
    exported.transform.translation[2] = 3;
    std::ostringstream csv, sdf;
    write_gpu_docking_pose_header(csv, true);
    write_gpu_docking_poses(csv, sdf, ligand, {exported}, 8, "ligand,\"one\"", true, &ligand);
    std::istringstream input(csv.str());
    std::string header, row;
    std::getline(input, header); std::getline(input, row);
    const auto columns = csv_fields(header), fields = csv_fields(row);
    require(columns.size() == 20 && fields.size() == columns.size(), "RMSD CSV shape differs");
    require(columns.back() == "rmsd_heavy_A", "RMSD header differs");
    require(fields[0] == "8" && fields[1] == "ligand,\"one\"" && fields[2] == "1" && fields[3] == "12",
            "CSV record metadata or escaping differs");
    close(std::stod(fields[4]), 4.25);
    close(std::stod(fields.back()), gpu_pose_heavy_atom_rmsd(ligand, exported.transform, ligand));
    const auto end = sdf.str().find("M  END");
    require(end != std::string::npos && sdf.str().find("$$$$") != std::string::npos,
            "SDF structure or record delimiter absent");
    std::unique_ptr<RDKit::RWMol> parsed(RDKit::MolBlockToMol(sdf.str().substr(0, end + 6), false, false));
    require(parsed && parsed->getNumAtoms() == ligand.getNumAtoms(), "SDF did not parse");
    const auto& conformer = parsed->getConformer();
    const std::vector<RDGeom::Point3D> expected = {{1, 3, 3}, {-1, 2, 3}, {-8, 11, 12}};
    for (unsigned int i = 0; i < expected.size(); ++i) {
        close(conformer.getAtomPos(i).x, expected[i].x, 1e-4);
        close(conformer.getAtomPos(i).y, expected[i].y, 1e-4);
        close(conformer.getAtomPos(i).z, expected[i].z, 1e-4);
    }
    for (const char* property : {"record_index", "rank", "alignment_index", "fast_weighted_score", "rmsd_heavy_A"})
        require(sdf.str().find(std::string("<") + property + ">") != std::string::npos,
                "SDF pose property absent");
    std::ostringstream missing_csv, missing_sdf;
    write_gpu_docking_pose_header(missing_csv, true);
    write_gpu_docking_poses(missing_csv, missing_sdf, ligand, {exported}, 1, "no reference", true);
    std::istringstream missing(missing_csv.str());
    std::getline(missing, header); std::getline(missing, row);
    require(csv_fields(row).size() == 20 && csv_fields(row).back().empty(), "Missing RMSD column was dropped");
    std::ostringstream plain_csv, plain_sdf;
    write_gpu_docking_pose_header(plain_csv);
    write_gpu_docking_poses(plain_csv, plain_sdf, ligand, {exported}, 1, "plain");
    std::istringstream plain(plain_csv.str());
    std::getline(plain, header); std::getline(plain, row);
    require(csv_fields(header).size() == 19 && csv_fields(row).size() == 19, "Plain CSV shape differs");
    require(header == "record_index,molecule_id,rank,alignment_index,fast_weighted_score,"
                      "clash_fraction,occupied_atoms,r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz",
            "Legacy Fast CSV header changed");
    require(plain_sdf.str().find("<rmsd_heavy_A>") == std::string::npos, "Unrequested RMSD property present");
    for (const char* property : {"pose_id", "weighted_interaction_score", "vinardo_like_score", "hybrid_score"})
        require(plain_sdf.str().find(std::string("<") + property + ">") == std::string::npos,
                "Legacy Fast SDF acquired a rescoring property");
    rejects([&] { write_gpu_docking_poses(plain_csv, plain_sdf, ligand, {exported}, 1, "bad", false, &ligand); });
    std::ostringstream invalid_csv, invalid_sdf;
    auto bad_reference = ligand;
    bad_reference.getAtomWithIdx(1)->setFormalCharge(-1);
    rejects([&] { write_gpu_docking_poses(invalid_csv, invalid_sdf, ligand, {exported}, 1, "bad", true, &bad_reference); });
    auto invalid = exported;
    invalid.fast_score.weighted_score = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { write_gpu_docking_poses(invalid_csv, invalid_sdf, ligand, {exported, invalid}, 1, "bad"); });
    require(invalid_csv.str().empty() && invalid_sdf.str().empty(), "Failed export wrote partial records");
    write_gpu_docking_poses(invalid_csv, invalid_sdf, ligand, {}, 1, "empty");
    require(invalid_csv.str().empty() && invalid_sdf.str().empty(), "Empty ranking produced records");
}

std::string sdf_property(const std::string& record, const std::string& property) {
    const auto name = record.find("<" + property + ">");
    require(name != std::string::npos, "Expected SDF property missing");
    const auto begin = record.find('\n', name);
    require(begin != std::string::npos, "Missing SDF property value");
    const auto end = record.find('\n', begin + 1);
    require(end != std::string::npos, "Unterminated SDF property value");
    return record.substr(begin + 1, end - begin - 1);
}

void test_extended_export() {
    auto ligand = molecule();
    // Inherited input properties must not masquerade as uncomputed scores.
    ligand.setProp("vinardo_like_score", std::string("stale"));
    ligand.setProp("hybrid_score", std::string("stale"));
    const auto first = pose(12, 4.25);
    auto second = pose(4, 5.5);
    second.transform.translation[0] = 10;
    std::vector<GpuDockingPoseScores> scores(2);
    scores[0].pose_id = 81;
    scores[0].weighted_interaction = WeightedInteractionScore{};
    scores[0].weighted_interaction->weighted_score = 10;
    scores[0].vinardo = VinardoLikeScore{-3, -4, 2, -0.5, -0.5};
    scores[0].vinardo_adjusted = -1;
    scores[0].hybrid = 0;
    scores[1].pose_id = 3;
    scores[1].weighted_interaction = WeightedInteractionScore{};
    scores[1].weighted_interaction->weighted_score = 8;

    std::ostringstream csv, sdf;
    write_gpu_docking_pose_header(csv, true, true);
    write_gpu_docking_poses(csv, sdf, ligand, {first, second}, 7, "rescored", true,
                            &ligand, &scores);
    std::istringstream input(csv.str());
    std::string header, row;
    std::getline(input, header);
    const auto columns = csv_fields(header);
    require(columns.size() == 29, "Extended RMSD header shape differs");
    require(columns[19] == "rmsd_heavy_A" && columns[20] == "pose_id" &&
            columns[21] == "weighted_interaction_score" && columns[22] == "vinardo_like_score" &&
            columns[23] == "vinardo_gaussian" && columns[24] == "vinardo_repulsion" &&
            columns[25] == "vinardo_hydrophobic" && columns[26] == "vinardo_hydrogen_bond" &&
            columns[27] == "vinardo_adjusted_score" && columns[28] == "hybrid_score",
            "Extended columns changed the legacy column positions or score order");
    std::getline(input, row);
    const auto first_fields = csv_fields(row);
    require(first_fields.size() == columns.size(), "Extended first row shape differs");
    require(first_fields[2] == "1" && first_fields[3] == "12" && first_fields[20] == "81",
            "Rank, alignment index or original pose ID were conflated");
    const std::vector<double> first_scores = {10, -3, -4, 2, -0.5, -0.5, -1, 0};
    for (std::size_t i = 0; i < first_scores.size(); ++i)
        close(std::stod(first_fields[21 + i]), first_scores[i]);
    std::getline(input, row);
    const auto second_fields = csv_fields(row);
    require(second_fields.size() == columns.size() && second_fields[2] == "2" &&
            second_fields[3] == "4" && second_fields[20] == "3",
            "Scores became detached from the ranked poses");
    close(std::stod(second_fields[16]), 10);
    close(std::stod(second_fields[19]), 10);
    close(std::stod(second_fields[21]), 8);
    for (std::size_t i = 22; i < second_fields.size(); ++i)
        require(second_fields[i].empty(), "Uncomputed score did not retain an empty CSV field");

    const auto delimiter = sdf.str().find("$$$$");
    require(delimiter != std::string::npos, "Missing extended SDF record");
    const auto first_record = sdf.str().substr(0, delimiter);
    const auto second_record = sdf.str().substr(delimiter + 4);
    require(sdf_property(first_record, "pose_id") == "81" &&
            sdf_property(first_record, "alignment_index") == "12" &&
            sdf_property(first_record, "rank") == "1",
            "Extended SDF identifiers changed");
    const std::vector<std::string> score_properties = {
        "weighted_interaction_score", "vinardo_like_score", "vinardo_gaussian", "vinardo_repulsion",
        "vinardo_hydrophobic", "vinardo_hydrogen_bond", "vinardo_adjusted_score", "hybrid_score"};
    for (std::size_t i = 0; i < score_properties.size(); ++i)
        close(std::stod(sdf_property(first_record, score_properties[i])), first_scores[i]);
    require(sdf_property(second_record, "pose_id") == "3" &&
            sdf_property(second_record, "alignment_index") == "4",
            "Second SDF score mapping differs");
    close(std::stod(sdf_property(second_record, "weighted_interaction_score")), 8);
    for (std::size_t i = 1; i < score_properties.size(); ++i)
        require(second_record.find("<" + score_properties[i] + ">") == std::string::npos,
                "Uncomputed or inherited score was exported into SDF");

    // Extended columns are independent of the optional RMSD column.
    std::ostringstream plain_header;
    write_gpu_docking_pose_header(plain_header, false, true);
    require(csv_fields(plain_header.str().substr(0, plain_header.str().size() - 1)).size() == 28,
            "Extended header without RMSD has a different shape");
    std::ostringstream invalid_csv, invalid_sdf;
    auto mismatched_scores = scores;
    mismatched_scores.pop_back();
    rejects([&] { write_gpu_docking_poses(invalid_csv, invalid_sdf, ligand,
        {first, second}, 1, "bad", false, nullptr, &mismatched_scores); });
    for (int field = 0; field < 4; ++field) {
        auto nonfinite = scores;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (field == 0) nonfinite[1].weighted_interaction->weighted_score = nan;
        if (field == 1) nonfinite[1].vinardo = VinardoLikeScore{0, 0, nan, 0, 0};
        if (field == 2) nonfinite[1].vinardo_adjusted = nan;
        if (field == 3) nonfinite[1].hybrid = nan;
        rejects([&] { write_gpu_docking_poses(invalid_csv, invalid_sdf, ligand,
            {first, second}, 1, "bad", false, nullptr, &nonfinite); });
    }
    require(invalid_csv.str().empty() && invalid_sdf.str().empty(),
            "Invalid rescoring values produced partial output records");
}

} // namespace

int main() {
    try {
        test_selection();
        test_rmsd();
        test_export();
        test_extended_export();
        std::cout << "GPU docking output tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "GPU docking output test failed: " << exception.what() << '\n';
        return 1;
    }
}
