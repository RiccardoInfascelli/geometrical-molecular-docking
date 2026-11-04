#include "gpu_docking_output.hpp"
#include "geometry.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SanitException.h>
#include <Geometry/point.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

void test_center_distance() {
    const auto ligand = molecule();
    const auto reference = heavy_atom_centroid(ligand);
    close(reference[0], 0.5);
    close(reference[1], 1.0);
    close(reference[2], 0.0);
    RigidTransform rigid;
    close(pose_center_distance(reference, rigid, reference), 0.0);
    rigid.translation = {3.0, 4.0, 0.0};
    close(pose_center_distance(reference, rigid, reference), 5.0);
    rigid.rotation = {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    rigid.translation = {1.5, 0.5, 0.0}; // Rotation about the heavy-atom centroid.
    close(pose_center_distance(reference, rigid, reference), 0.0);
    rigid.translation = {2.0, -3.0, 1.0};
    auto transformed = ligand;
    for (unsigned int i = 0; i < transformed.getNumAtoms(); ++i) {
        const auto point = ligand.getConformer().getAtomPos(i);
        transformed.getConformer().setAtomPos(i, {-point.y + 2.0, point.x - 3.0, point.z + 1.0});
    }
    const auto actual_center = heavy_atom_centroid(transformed);
    close(pose_center_distance(reference, rigid, reference),
        std::hypot(actual_center[0] - reference[0], actual_center[1] - reference[1], actual_center[2] - reference[2]));
    close(pose_center_distance(reference, rigid, reference), std::sqrt(13.5));
    auto hydrogen_changed = ligand;
    hydrogen_changed.getConformer().setAtomPos(2, {1000, -500, 800});
    require(heavy_atom_centroid(hydrogen_changed) == reference, "Hydrogen moved the heavy-atom centroid");

    RDKit::RWMol missing;
    rejects([&] { heavy_atom_centroid(missing); });
    auto no_coordinates = ligand;
    no_coordinates.clearConformers();
    rejects([&] { heavy_atom_centroid(no_coordinates); });
    RDKit::RWMol hydrogen_only;
    hydrogen_only.addAtom(new RDKit::Atom(1), true, true);
    hydrogen_only.addConformer(new RDKit::Conformer(1), true);
    rejects([&] { heavy_atom_centroid(hydrogen_only); });
    auto invalid = ligand;
    invalid.getConformer().setAtomPos(0, {std::numeric_limits<double>::quiet_NaN(), 0, 0});
    rejects([&] { heavy_atom_centroid(invalid); });
    rigid.translation[0] = std::numeric_limits<double>::infinity();
    rejects([&] { pose_center_distance(reference, rigid, reference); });
    rejects([&] { pose_center_distance(reference, RigidTransform{},
        {std::numeric_limits<double>::quiet_NaN(), 0, 0}); });

    auto selected = pose(11, 7.0);
    selected.transform.translation[0] = 3.0;
    selected.transform.translation[1] = 4.0;
    const std::vector<GpuPipelinePose> ranked{selected, pose(12, 6.0)};
    const auto values = gpu_docking_pose_center_distances(ligand, ranked, reference, 2);
    close(values[0], 5.0);
    close(values[1], 0.0);
    RDKit::RWMol unrelated_reference;
    unrelated_reference.addAtom(new RDKit::Atom(7), true, true);
    auto* reference_coordinates = new RDKit::Conformer(1);
    reference_coordinates->setAtomPos(0, {0.5, 1.0, 0.0});
    unrelated_reference.addConformer(reference_coordinates, true);
    const auto unrelated = gpu_docking_pose_center_distances(ligand, ranked,
        heavy_atom_centroid(unrelated_reference), 2);
    require(unrelated == values, "Spatial centroid distance incorrectly required a chemical atom mapping");
    rejects([&] { gpu_docking_pose_center_distances(ligand, ranked, reference, 3); });
    rejects([&] { gpu_docking_pose_center_distances(ligand, {},
        {std::numeric_limits<double>::quiet_NaN(), 0, 0}, 0); });
    auto bad_pose = selected;
    bad_pose.transform.valid = 0;
    rejects([&] { gpu_docking_pose_center_distances(ligand, {bad_pose}, reference, 1); });
    for (auto mode : {PoseExportMode::TopK, PoseExportMode::All})
        for (auto detail : {PoseExportDetail::Compact, PoseExportDetail::Full}) {
            PoseExportOptions options{mode, 1, detail};
            std::vector<GpuDockingPoseScores> scores(2);
            scores[0].pose_id = 21;
            scores[1].pose_id = 22;
            const auto count = pose_export_count(options, ranked.size());
            const auto cached = gpu_docking_pose_center_distances(ligand, ranked, reference, count);
            std::ostringstream off, on;
            write_gpu_docking_pose_header(off, true, true, &options);
            write_gpu_docking_poses(off, nullptr, ligand, ranked, 1, "center test", true,
                &ligand, &scores, &options);
            write_gpu_docking_pose_header(on, true, true, &options, true);
            write_gpu_docking_poses(on, nullptr, ligand, ranked, 1, "center test", true,
                &ligand, &scores, &options, nullptr, &cached);
            std::istringstream before(off.str()), after(on.str());
            std::string old_line, new_line;
            for (std::size_t row = 0; std::getline(before, old_line); ++row) {
                require(bool(std::getline(after, new_line)), "Center diagnostic omitted an output row");
                const auto separator = new_line.rfind(',');
                require(separator != std::string::npos && new_line.substr(0, separator) == old_line,
                        "Center diagnostic changed existing pose, score, rank or RMSD columns");
                if (!row) require(new_line.substr(separator + 1) == "center_distance_A", "Wrong center column");
                else close(std::stod(new_line.substr(separator + 1)), cached.at(row - 1));
            }
            require(!std::getline(after, new_line), "Center diagnostic added a pose");
        }
    std::ostringstream invalid_output;
    for (const auto& bad : std::vector<std::vector<double>>{{1.0}, {-1.0, 0.0},
             {std::numeric_limits<double>::quiet_NaN(), 0.0}})
        rejects([&] { write_gpu_docking_poses(invalid_output, nullptr, ligand, ranked, 1,
            "invalid", false, nullptr, nullptr, nullptr, nullptr, &bad); });
    require(invalid_output.str().empty(), "Invalid center cache produced partial CSV records");
    std::ostringstream csv, sdf;
    write_gpu_docking_poses(csv, sdf, ligand, ranked, 1, "centers", false,
        nullptr, nullptr, nullptr, nullptr, &values);
    require(sdf.str().find("<center_distance_A>") != std::string::npos,
            "Selected pose SDF lost its requested center distance");
    require(ranked[0].fast_score.weighted_score == 7.0 && ranked[0].alignment_index == 11 &&
        ranked[0].transform.translation[0] == 3.0, "Center diagnostic mutated the scientific pose");
}

void test_summary_csv_escaping() {
    const std::string online_scope =
        "input read through final record export and producer join; excludes receptor startup, timeline and summary writes";
    const std::string query_scope =
        "host query packing in producer; search_completed_s is query work plus completed CUDA wrapper work, not pipeline elapsed time";
    std::ostringstream summary;
    write_gpu_docking_summary_entry(summary, "online_scope", online_scope);
    write_gpu_docking_summary_entry(summary, "query_preparation_scope", query_scope);
    require(summary.str() == "online_scope,\"" + online_scope + "\"\nquery_preparation_scope,\"" + query_scope + "\"\n",
            "Overlap descriptions were not escaped as two-column CSV rows");

    const std::vector<std::pair<std::string, std::string>> cases{
        {"online_scope", online_scope}, {"query_preparation_scope", query_scope},
        {"protein", "/input/target,\"quoted\".pdb"},
        {"DOCKING_QUERY_METHOD", "value,with \"quotes\""},
        {"literal_quotes", "\"already quoted\""},
        {"reference_mol2", ""}, {"processed", "1000"},
        {"key,\"quoted\"", "first line\r\nsecond line"}};
    for (const auto& [key, value] : cases) {
        std::ostringstream csv;
        write_gpu_docking_summary_entry(csv, key, value);
        const auto encoded = csv.str();
        require(!encoded.empty() && encoded.back() == '\n', "Summary CSV record terminator is missing");
        const auto decoded = csv_fields(encoded.substr(0, encoded.size() - 1));
        require(decoded.size() == 2 && decoded[0] == key && decoded[1] == value,
                "Summary CSV lost a field or escaped a raw value more than once");
    }
    std::ostringstream quoted;
    write_gpu_docking_summary_entry(quoted, "path", "target,\"quoted\".pdb");
    require(quoted.str() == "path,\"target,\"\"quoted\"\".pdb\"\n",
            "Summary CSV did not double literal quotes within a quoted field");
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

void test_csv_only_export() {
    const auto ligand = molecule();
    const std::vector<GpuPipelinePose> poses = {pose(3, 7.0), pose(5, 4.0)};
    std::vector<GpuDockingPoseScores> scores(poses.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        scores[i].pose_id = i + 9;
        scores[i].weighted_interaction.emplace();
        scores[i].weighted_interaction->weighted_score = 2.0 + i;
        scores[i].vinardo.emplace();
        scores[i].vinardo->total = -3.0 - i;
        scores[i].vinardo_adjusted = -1.0 - i;
        scores[i].hybrid = 0.3 + i;
    }
    std::ostringstream full_csv, sdf, csv_only;
    write_gpu_docking_poses(full_csv, sdf, ligand, poses, 4, "same", true, &ligand, &scores);
    write_gpu_docking_poses(csv_only, nullptr, ligand, poses, 4, "same", true, &ligand, &scores);
    require(full_csv.str() == csv_only.str(), "Disabling SDF changed CSV scores, RMSD or pose order");
    require(!sdf.str().empty(), "Default SDF export was disabled");

    // A five-carbon aromatic ring cannot be kekulized. CSV export must not
    // invoke the SDF writer or alter this deliberately invalid test molecule.
    RDKit::RWMol aromatic;
    for (unsigned int i = 0; i < 5; ++i) {
        auto* atom = new RDKit::Atom(6);
        atom->setIsAromatic(true);
        aromatic.addAtom(atom, true, true);
    }
    for (unsigned int i = 0; i < 5; ++i) aromatic.addBond(i, (i + 1) % 5, RDKit::Bond::AROMATIC);
    aromatic.addConformer(new RDKit::Conformer(5), true);
    aromatic.updatePropertyCache(false);
    std::ostringstream failed_csv, failed_sdf;
    bool rejected = false;
    try { write_gpu_docking_poses(failed_csv, failed_sdf, aromatic, {pose(1, 2)}, 1, "aromatic"); }
    catch (const RDKit::KekulizeException&) { rejected = true; }
    require(rejected, "The aromatic fixture no longer exercises SDF kekulization failure");
    require(failed_csv.str().empty() && failed_sdf.str().empty(), "Failed SDF export committed partial data");
    write_gpu_docking_poses(failed_csv, nullptr, aromatic, {pose(1, 2)}, 1, "aromatic");
    require(!failed_csv.str().empty(), "CSV-only export still required kekulization");
    for (const auto* bond : aromatic.bonds())
        require(bond->getBondType() == RDKit::Bond::AROMATIC, "Export changed input bond orders");
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

void test_pose_output_controls() {
    const auto ligand = molecule();
    std::vector<GpuPipelinePose> ranked = {pose(8, 3), pose(2, 2), pose(4, 1)};
    for (std::size_t i = 0; i < ranked.size(); ++i)
        ranked[i].transform.translation[0] = 3.0 - i;
    const auto rmsds = gpu_docking_pose_rmsds(ligand, ranked, ligand);
    require(rmsds.size() == ranked.size(), "Ranked-pool RMSD coverage differs");
    for (std::size_t i = 0; i < rmsds.size(); ++i) close(rmsds[i], 3.0 - i);

    PoseExportOptions options;
    options.top_k = 1;
    std::ostringstream original, controlled;
    write_gpu_docking_poses(original, nullptr, ligand, {ranked[0]}, 1, "fixed", true, &ligand);
    write_gpu_docking_poses(controlled, nullptr, ligand, ranked, 1, "fixed", true, &ligand,
                            nullptr, &options, &rmsds);
    require(original.str() == controlled.str(), "Top-k export changed the selected pose or cached RMSD");
    options.mode = PoseExportMode::All;
    std::ostringstream all, legacy;
    write_gpu_docking_poses(all, nullptr, ligand, ranked, 1, "fixed", true, &ligand,
                            nullptr, &options, &rmsds);
    write_gpu_docking_poses(legacy, nullptr, ligand, ranked, 1, "fixed", true, &ligand);
    require(all.str() == legacy.str(), "All export changed ranked pose identities or score precision");

    options.mode = PoseExportMode::None;
    std::ostringstream none, no_sdf;
    auto unavailable_coordinates = ligand;
    unavailable_coordinates.clearConformers();
    write_gpu_docking_pose_header(none, true, true, &options);
    write_gpu_docking_poses(none, no_sdf, unavailable_coordinates, ranked, 1, "fixed", true,
                            nullptr, nullptr, &options);
    require(none.str().empty() && no_sdf.str().empty(),
            "None export formatted CSV/SDF or required coordinate serialization");
    require(ranked[0].alignment_index == 8 && ranked[1].alignment_index == 2 && rmsds[0] == 3,
            "Output policy reordered poses using RMSD");

    options.mode = PoseExportMode::TopK;
    options.detail = PoseExportDetail::Compact;
    std::vector<GpuDockingPoseScores> scores(ranked.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        scores[i].pose_id = i + 40;
        scores[i].weighted_interaction.emplace();
        scores[i].weighted_interaction->weighted_score = 5.0 - i;
        scores[i].vinardo.emplace();
        scores[i].vinardo->total = -2.0 - i;
        scores[i].vinardo_adjusted = -3.0 - i;
        scores[i].hybrid = 1.0 - i;
    }
    std::ostringstream compact;
    write_gpu_docking_pose_header(compact, true, true, &options);
    write_gpu_docking_poses(compact, nullptr, ligand, ranked, 1, "fixed", true, &ligand,
                            &scores, &options, &rmsds);
    std::istringstream rows(compact.str());
    std::string header, row, extra;
    std::getline(rows, header); std::getline(rows, row); std::getline(rows, extra);
    const auto names = csv_fields(header), fields = csv_fields(row);
    require(names.size() == 13 && fields.size() == names.size() && extra.empty(),
            "Compact export shape or top-k count differs");
    require(header.find("r00") == std::string::npos && header.find("vinardo_gaussian") == std::string::npos &&
            names[7] == "rmsd_heavy_A" && names[8] == "pose_id" && names[12] == "hybrid_score" &&
            fields[3] == "8" && fields[8] == "40" && fields[7] == "3",
            "Compact export lost pose identity, scores or RMSD");

    options.detail = PoseExportDetail::Full;
    std::ostringstream full_header, old_header;
    write_gpu_docking_pose_header(full_header, true, true, &options);
    write_gpu_docking_pose_header(old_header, true, true);
    require(full_header.str() == old_header.str(), "Explicit full output changed the legacy schema");
    std::vector<double> bad_rmsds{1};
    std::ostringstream invalid;
    rejects([&] { write_gpu_docking_poses(invalid, nullptr, ligand, ranked, 1, "bad", true,
                                         &ligand, &scores, &options, &bad_rmsds); });
    bad_rmsds = rmsds; bad_rmsds[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { write_gpu_docking_poses(invalid, nullptr, ligand, ranked, 1, "bad", true,
                                         &ligand, &scores, &options, &bad_rmsds); });
    require(invalid.str().empty(), "Invalid cached RMSD produced partial output");
}

void test_ranked_pool_metrics() {
    std::vector<double> values(25, 4.0);
    values[0] = 3.0; values[4] = 2.0; values[9] = 1.5;
    values[19] = 1.0; values[24] = .25;
    std::ostringstream metrics;
    write_gpu_docking_pose_metrics_header(metrics);
    write_gpu_docking_pose_metrics(metrics, values, 2, "pool");
    write_gpu_docking_pose_metrics(metrics, {}, 3, "empty");
    std::istringstream rows(metrics.str());
    std::string header, populated, empty;
    std::getline(rows, header); std::getline(rows, populated); std::getline(rows, empty);
    const auto fields = csv_fields(populated), missing = csv_fields(empty);
    require(csv_fields(header).size() == 14 && fields.size() == 14 && missing.size() == 14,
            "Pool metric schema differs");
    require(fields[2] == "25" && fields[9] == "3" && fields[10] == "2" &&
            fields[11] == "1.5" && fields[12] == "1" && fields[13] == "0.25",
            "Pool metrics used export limits or RMSD-based selection");
    require(missing[2] == "0" && std::all_of(missing.begin() + 3, missing.end(),
            [](const std::string& field) { return field.empty(); }), "Empty pool RMSDs became zeros");
    auto selected = pose(17, 9);
    GpuDockingPoseScores scores;
    scores.pose_id = 4;
    scores.hybrid = 2;
    scores.vinardo_adjusted = -7;
    scores.weighted_interaction.emplace();
    scores.weighted_interaction->weighted_score = 6;
    std::ostringstream no_reference;
    write_gpu_docking_pose_metrics(no_reference, {}, 1, "no reference", 25, &selected, &scores);
    const auto decision = csv_fields(no_reference.str().substr(0, no_reference.str().size() - 1));
    require(decision[2] == "25" && decision[3] == "17" && decision[4] == "4" &&
            decision[5] == "9" && decision[6] == "6" && decision[7] == "-7" && decision[8] == "2" &&
            std::all_of(decision.begin() + 9, decision.end(), [](const std::string& value) { return value.empty(); }),
            "None/reference-free metrics lost the scientific Top-1 decision or invented RMSD");
    std::ostringstream invalid;
    for (double bad : {-1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
        rejects([&] { write_gpu_docking_pose_metrics(invalid, {bad}, 1, "bad"); });
    rejects([&] { write_gpu_docking_pose_metrics(invalid, {1.0}, 1, "truncated", 25); });
    require(invalid.str().empty(), "Invalid pool metrics produced partial records");
}

} // namespace

int main() {
    try {
        test_summary_csv_escaping();
        test_selection();
        test_rmsd();
        test_csv_only_export();
        test_export();
        test_extended_export();
        test_pose_output_controls();
        test_ranked_pool_metrics();
        test_center_distance();
        std::cout << "GPU docking output tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "GPU docking output test failed: " << exception.what() << '\n';
        return 1;
    }
}
