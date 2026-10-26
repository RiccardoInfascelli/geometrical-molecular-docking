#include "pose_rescoring.hpp"

#include "geometry.hpp"
#include "interaction_scoring.hpp"

#include <GraphMol/Atom.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/RWMol.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool close(double a, double b)
{
    return std::abs(a - b) <= 1e-12 * (1.0 + std::abs(b));
}

RDKit::RWMol make_carbons(const std::vector<RDGeom::Point3D>& positions)
{
    RDKit::RWMol molecule;
    auto conformer = std::make_unique<RDKit::Conformer>(
        static_cast<unsigned int>(positions.size()));
    for (unsigned int i = 0; i < positions.size(); ++i) {
        molecule.addAtom(new RDKit::Atom(6), true, true);
        conformer->setAtomPos(i, positions[i]);
    }
    molecule.addConformer(conformer.release(), true);
    return molecule;
}

interaction::interaction_area hydrophobic_area(float x)
{
    interaction::interaction_area area{};
    area.pattern_class = interaction::interaction_class_pattern::Hydrophobic;
    area.pattern = interaction::interaction_smarts_pattern::Hydrophobic;
    area.num_atoms = 1;
    area.centroid = {x, 0.25, 0.25};
    area.x[0] = x;
    area.y[0] = area.z[0] = 0.25;
    return area;
}

struct Fixture
{
    RDKit::RWMol ligand = make_carbons({{0.25, 0.25, 0.25}, {1.25, 0.25, 0.25}});
    RDKit::RWMol receptor = make_carbons({{3.25, 0.25, 0.25}});
    VinardoLikeScorer vinardo{receptor};
    PreparedLigand prepared;
    ProteinGrid grid;
    std::vector<interaction::interaction_area> protein_areas{hydrophobic_area(3.25)};
    std::vector<RankedPoseDiagnostic> poses{5};

    Fixture()
    {
        prepared.areas = {hydrophobic_area(0.25), hydrophobic_area(1.25)};
        prepared.matching_areas = build_ligand_matching_areas(prepared.areas);
        grid.voxel_size = 1.0;
        grid.nx = 20;
        grid.ny = grid.nz = 1;
        grid.voxels.resize(20);
        for (int voxel : {0, 1, 3}) {
            add_interaction(grid.voxels[voxel].interactions, InteractionKey::Hydrophobic);
        }
        const std::array<double, 5> translations{8.0, 0.0, 2.0, 0.0, 14.0};
        for (std::size_t i = 0; i < poses.size(); ++i) {
            poses[i].transform.translation[0] = translations[i];
            poses[i].fast_weighted_score = 1000.0 + static_cast<double>(i);
            poses[i].weighted_interaction_score.weighted_score = -1000.0;
            poses[i].occupied_atoms = i;
            poses[i].clash_fraction = 0.01 * static_cast<double>(i);
            poses[i].rmsd = std::numeric_limits<double>::quiet_NaN();
        }
        poses[0].rmsd = 0.0;
        poses[1].rmsd = 100.0;
        poses[3].rmsd = -1.0;
    }

    PoseRescoringResult score(
        const std::vector<RankedPoseDiagnostic>& input,
        std::optional<std::size_t> top_n = std::nullopt,
        bool reuse_fast = false,
        PoseRescoringMethods methods = {}) const
    {
        return rescore_poses(input, ligand, prepared, grid, protein_areas,
                             vinardo, top_n, reuse_fast, methods);
    }
};

void require_weighted_interaction_equal(const WeightedInteractionScore& a, const WeightedInteractionScore& b)
{
    require(a.total_interactions == b.total_interactions &&
            a.satisfied_ligand_areas == b.satisfied_ligand_areas &&
            a.satisfied_protein_areas == b.satisfied_protein_areas &&
            a.hydrophobic == b.hydrophobic && a.hydrogen_bonds == b.hydrogen_bonds &&
            a.halogen_bonds == b.halogen_bonds && a.ionic == b.ionic &&
            a.cation_pi == b.cation_pi && a.pi_stacking == b.pi_stacking &&
            a.metal == b.metal && close(a.weighted_score, b.weighted_score),
            "WeightedInteraction scores must belong to the source pose ID");
}

void require_vinardo_equal(const VinardoLikeScore& a, const VinardoLikeScore& b)
{
    require(close(a.total, b.total) && close(a.gaussian, b.gaussian) &&
            close(a.repulsion, b.repulsion) && close(a.hydrophobic, b.hydrophobic) &&
            close(a.hydrogen_bond, b.hydrogen_bond),
            "All Vinardo-like components must belong to the source pose ID");
}

std::vector<std::size_t> pose_ids(const PoseRescoringResult& result)
{
    std::vector<std::size_t> ids;
    for (const auto& row : result.poses) ids.push_back(row.pose_id);
    return ids;
}

void test_same_pose_scores(const Fixture& fixture)
{
    const auto all = fixture.score(fixture.poses);
    require(all.weighted_interaction_calculated && all.vinardo_calculated,
            "Default rescoring must still calculate both methods");
    require(all.input_pose_count == fixture.poses.size() &&
            pose_ids(all) == std::vector<std::size_t>{0, 1, 2, 3, 4},
            "No prefilter must preserve every survivor and its original order");
    const std::array<double, 5> expected_fast{0.0, 2.0, 1.0, 2.0, 0.0};
    for (const auto& row : all.poses) {
        const auto& source = fixture.poses[row.pose_id];
        require(row.pose.transform.rotation == source.transform.rotation &&
                row.pose.transform.translation == source.transform.translation &&
                row.pose.occupied_atoms == source.occupied_atoms &&
                row.pose.clash_fraction == source.clash_fraction,
                "Rescoring must preserve transforms and clash data");
        const auto fast = score_matching_areas_weighted(
            transform_matching_areas(fixture.prepared.matching_areas, source.transform),
            fixture.grid);
        require(row.pose.fast_weighted_score == expected_fast[row.pose_id] &&
                row.pose.fast_weighted_score == fast.weighted_score,
                "Fast scores must be recomputed for the same source transforms");
        require_weighted_interaction_equal(row.pose.weighted_interaction_score, score_weighted_interactions(
            fixture.protein_areas,
            transform_interaction_areas(fixture.prepared.areas, source.transform)));
        require_vinardo_equal(row.vinardo, fixture.vinardo.score(fixture.ligand, source.transform));
        require(source.fast_weighted_score == 1000.0 + static_cast<double>(row.pose_id) &&
                source.weighted_interaction_score.weighted_score == -1000.0,
                "Rescoring must not modify the original survivor list");
    }
    require(all.poses[1].pose.weighted_interaction_score.weighted_score > 0.0 &&
            all.poses[4].pose.weighted_interaction_score.weighted_score == 0.0 &&
            all.poses[2].vinardo.total != all.poses[4].vinardo.total,
            "Synthetic chemistry must exercise nonconstant WeightedInteraction and Vinardo-like scores");

    const auto selected = fixture.score(fixture.poses, 3);
    require(selected.input_pose_count == 5 &&
            pose_ids(selected) == std::vector<std::size_t>{1, 2, 3} &&
            pose_ids(fixture.score(fixture.poses, 1)) == std::vector<std::size_t>{1},
            "Fast top-N must break ties by ID and export in original survivor order");
    for (const auto& row : selected.poses) {
        const auto& expected = all.poses[row.pose_id];
        require(row.pose.fast_weighted_score == expected.pose.fast_weighted_score,
                "Selection must not change Fast scores or pose IDs");
        require_weighted_interaction_equal(row.pose.weighted_interaction_score, expected.pose.weighted_interaction_score);
        require_vinardo_equal(row.vinardo, expected.vinardo);
    }
    require(pose_ids(fixture.score(fixture.poses, 99)) == pose_ids(all),
            "A prefilter larger than the pool must retain all survivors");

    auto changed_rmsd = fixture.poses;
    for (std::size_t i = 0; i < changed_rmsd.size(); ++i) {
        changed_rmsd[i].rmsd = static_cast<double>(i);
    }
    changed_rmsd[1].rmsd = std::numeric_limits<double>::infinity();
    require(pose_ids(fixture.score(changed_rmsd, 3)) == pose_ids(selected),
            "RMSD, including nonfinite values, must not affect selection");
}

void test_precomputed_fast_and_errors(const Fixture& fixture)
{
    auto poses = fixture.poses;
    const std::array<double, 5> scores{9.0, -4.0, 9.0, 1.0, -8.0};
    for (std::size_t i = 0; i < poses.size(); ++i) poses[i].fast_weighted_score = scores[i];
    const auto all = fixture.score(poses, std::nullopt, true);
    require(pose_ids(all) == std::vector<std::size_t>{0, 1, 2, 3, 4} &&
            all.fast_scoring_seconds == 0.0,
            "GPU Fast reuse must not recompute or filter the survivor pool");
    for (const auto& row : all.poses) {
        require(row.pose.fast_weighted_score == scores[row.pose_id],
                "Finite precomputed Fast values, including negatives, must be preserved");
    }
    require(pose_ids(fixture.score(poses, 1, true)) == std::vector<std::size_t>{0} &&
            pose_ids(fixture.score(poses, 3, true)) == std::vector<std::size_t>{0, 2, 3},
            "Precomputed Fast selection must use score then ID, not WeightedInteraction or Vinardo");

    bool rejected = false;
    try { fixture.score(poses, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Zero is not a valid explicit Fast top-N");
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
        poses[4].fast_weighted_score = invalid;
        rejected = false;
        try { fixture.score(poses, 1, true); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "The experiment must report nonfinite Fast even outside top-N");
    }
    const auto empty = fixture.score({});
    require(empty.input_pose_count == 0 && empty.poses.empty(),
            "An empty survivor pool must remain empty");
}

void test_requested_methods(const Fixture& fixture)
{
    const auto both = fixture.score(fixture.poses, 3, true);
    const auto weighted_interaction = fixture.score(fixture.poses, 3, true, {true, false});
    const auto vinardo = fixture.score(fixture.poses, 3, true, {false, true});
    const auto neither = fixture.score(fixture.poses, 3, true, {false, false});
    require(weighted_interaction.weighted_interaction_calculated && !weighted_interaction.vinardo_calculated &&
            !vinardo.weighted_interaction_calculated && vinardo.vinardo_calculated &&
            !neither.weighted_interaction_calculated && !neither.vinardo_calculated,
            "Rescoring flags must identify requested calculations");
    require(weighted_interaction.vinardo_scoring_seconds == 0.0 && vinardo.weighted_interaction_scoring_seconds == 0.0 &&
            neither.weighted_interaction_scoring_seconds == 0.0 && neither.vinardo_scoring_seconds == 0.0,
            "Unrequested scoring loops must have no measured work");
    require(pose_ids(weighted_interaction) == pose_ids(both) && pose_ids(vinardo) == pose_ids(both) &&
            pose_ids(neither) == pose_ids(both),
            "Requested scorer must not change the Fast prefilter pool");
    for (std::size_t i = 0; i < both.poses.size(); ++i) {
        require_weighted_interaction_equal(weighted_interaction.poses[i].pose.weighted_interaction_score, both.poses[i].pose.weighted_interaction_score);
        require_vinardo_equal(vinardo.poses[i].vinardo, both.poses[i].vinardo);
        require(neither.poses[i].pose.fast_weighted_score == both.poses[i].pose.fast_weighted_score,
                "Disabling rescoring must preserve precomputed Fast");
    }

    // An unsupported ligand element is a Vinardo error. WeightedInteraction-only must not
    // invoke that scorer; the default joint path must still reject it.
    auto unsupported = fixture.ligand;
    unsupported.getAtomWithIdx(0)->setAtomicNum(30);
    const auto weighted_interaction_only = rescore_poses(fixture.poses, unsupported, fixture.prepared,
        fixture.grid, fixture.protein_areas, fixture.vinardo, std::nullopt, true, {true, false});
    require(weighted_interaction_only.poses.size() == fixture.poses.size(), "WeightedInteraction-only invoked Vinardo typing");
    bool rejected = false;
    try {
        rescore_poses(fixture.poses, unsupported, fixture.prepared, fixture.grid,
                     fixture.protein_areas, fixture.vinardo, std::nullopt, true);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Default joint path must still report unsupported Vinardo elements");
    const auto empty = fixture.score({}, std::nullopt, true, {true, false});
    require(empty.poses.empty() && empty.weighted_interaction_calculated && !empty.vinardo_calculated,
            "Empty pool must preserve requested calculation flags");
}

std::vector<std::string> csv_fields(const std::string& line)
{
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

void test_csv(const Fixture& fixture)
{
    const auto result = fixture.score(fixture.poses, 3);
    std::ostringstream output;
    write_pose_scores_csv(output, result, 7);
    std::istringstream input(output.str());
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "CSV header is missing");
    const auto header = csv_fields(line);
    require(header.size() == 26 && header[0] == "record_index" && header[1] == "pose_id" &&
            header[2] == "fast_score" && header[3] == "weighted_interaction_score" &&
            header[4] == "vinardo_like_score" && header[23] == "tx" &&
            line.find("rmsd") == std::string::npos && line.find("rank") == std::string::npos,
            "Joint CSV must identify all scores and transforms without RMSD or ranking");
    std::size_t rows = 0;
    while (std::getline(input, line)) {
        const auto fields = csv_fields(line);
        require(rows < result.poses.size() && fields.size() == header.size(),
                "Every pose CSV row must match the header");
        const auto& row = result.poses[rows];
        require(std::stoull(fields[0]) == 7 && std::stoull(fields[1]) == row.pose_id &&
                close(std::stod(fields[2]), row.pose.fast_weighted_score) &&
                close(std::stod(fields[3]), row.pose.weighted_interaction_score.weighted_score) &&
                close(std::stod(fields[4]), row.vinardo.total),
                "CSV scores must remain associated with their stable pose ID");
        for (std::size_t i = 0; i < 9; ++i) {
            require(close(std::stod(fields[14 + i]), row.pose.transform.rotation[i]),
                    "CSV rotation must belong to the scored pose");
        }
        for (std::size_t i = 0; i < 3; ++i) {
            require(close(std::stod(fields[23 + i]), row.pose.transform.translation[i]),
                    "CSV translation must belong to the scored pose");
        }
        ++rows;
    }
    require(rows == result.poses.size(), "CSV must export every selected pose exactly once");
    std::ostringstream empty;
    write_pose_scores_csv(empty, fixture.score({}), 7);
    require(empty.str() == output.str().substr(0, output.str().find('\n') + 1),
            "An empty pool must still produce the same CSV header");
}

} // namespace

int main()
{
    try {
        const Fixture fixture;
        test_same_pose_scores(fixture);
        test_precomputed_fast_and_errors(fixture);
        test_requested_methods(fixture);
        test_csv(fixture);
        std::cout << "Joint pose rescoring tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
