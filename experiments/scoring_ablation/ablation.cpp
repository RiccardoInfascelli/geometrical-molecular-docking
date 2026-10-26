#include "ablation.hpp"

#include <charconv>
#include <cmath>
#include <istream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace scoring_ablation {
namespace {

bool read_row(std::istream& input, std::vector<std::string>& fields)
{
    fields.assign(1, {});
    bool quoted = false, closed_quote = false, seen = false;
    char character;
    while (input.get(character)) {
        seen = true;
        if (quoted) {
            if (character == '"') {
                if (input.peek() == '"') {
                    input.get();
                    fields.back() += '"';
                } else {
                    quoted = false;
                    closed_quote = true;
                }
            } else fields.back() += character;
        } else if (character == ',') {
            fields.emplace_back();
            closed_quote = false;
        } else if (character == '\n' || character == '\r') {
            if (character == '\r' && input.peek() == '\n') input.get();
            return true;
        } else if (character == '"' && fields.back().empty() && !closed_quote) {
            quoted = true;
        } else {
            if (closed_quote || character == '"')
                throw std::runtime_error("Malformed quoted CSV field");
            fields.back() += character;
        }
    }
    if (input.bad()) throw std::runtime_error("Could not read pose CSV");
    if (quoted) throw std::runtime_error("Unclosed quoted CSV field");
    return seen;
}

std::size_t integer(const std::string& value, const std::string& column)
{
    std::size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::runtime_error("Invalid integer in " + column);
    return result;
}

double number(const std::string& value, const std::string& column)
{
    std::size_t end = 0;
    double result;
    try { result = std::stod(value, &end); }
    catch (const std::exception&) { throw std::runtime_error("Invalid number in " + column); }
    if (end != value.size() || !std::isfinite(result))
        throw std::runtime_error("Nonfinite or invalid number in " + column);
    return result;
}

} // namespace

const std::array<WeightedInteractionFamily, 7>& weighted_interaction_families()
{
    static const std::array<WeightedInteractionFamily, 7> families{{
        {"hydrophobic", &WeightedInteractionWeights::hydrophobic, &WeightedInteractionScore::hydrophobic},
        {"pi_stacking", &WeightedInteractionWeights::pi_stacking, &WeightedInteractionScore::pi_stacking},
        {"hydrogen_bond", &WeightedInteractionWeights::hydrogen_bond, &WeightedInteractionScore::hydrogen_bonds},
        {"halogen_bond", &WeightedInteractionWeights::halogen_bond, &WeightedInteractionScore::halogen_bonds},
        {"cation_pi", &WeightedInteractionWeights::cation_pi, &WeightedInteractionScore::cation_pi},
        {"ionic", &WeightedInteractionWeights::ionic, &WeightedInteractionScore::ionic},
        {"metal", &WeightedInteractionWeights::metal, &WeightedInteractionScore::metal}
    }};
    return families;
}

const std::array<const char*, 4>& vinardo_terms()
{
    static const std::array<const char*, 4> terms{
        "gaussian", "repulsion", "hydrophobic", "hydrogen_bond"};
    return terms;
}

WeightedInteractionWeights weighted_interaction_weights_without(std::size_t family)
{
    WeightedInteractionWeights weights;
    weights.*(weighted_interaction_families().at(family).weight) = 0.0;
    return weights;
}

VinardoLikeScore vinardo_without(const VinardoLikeScore& score, std::size_t term)
{
    VinardoLikeScore result = score;
    switch (term) {
    case 0: result.gaussian = 0.0; break;
    case 1: result.repulsion = 0.0; break;
    case 2: result.hydrophobic = 0.0; break;
    case 3: result.hydrogen_bond = 0.0; break;
    default: throw std::out_of_range("Unknown Vinardo term");
    }
    // Use the scorer's sum order, retaining small terms after a large one is removed.
    result.total = result.gaussian + result.repulsion + result.hydrophobic + result.hydrogen_bond;
    return result;
}

std::vector<InputPose> read_pose_pool(std::istream& input, std::size_t record)
{
    if (record == 0) throw std::invalid_argument("Record indices are one-based");
    std::vector<std::string> header, fields;
    if (!read_row(input, header)) throw std::runtime_error("Pose CSV has no header");
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (!columns.emplace(header[i], i).second)
            throw std::runtime_error("Duplicate CSV column: " + header[i]);
    }
    const auto required = [&](const std::string& column) {
        const auto found = columns.find(column);
        if (found == columns.end()) throw std::runtime_error("Missing CSV column: " + column);
        return found->second;
    };
    const auto record_column = required("record_index");
    const auto id_column = required(columns.count("pose_id") ? "pose_id" : "alignment_index");
    const auto fast_column = required(columns.count("fast_score") ? "fast_score" : "fast_weighted_score");
    const auto clash_column = required("clash_fraction");
    const auto occupied_column = required("occupied_atoms");
    const std::array<const char*, 12> transform_names{
        "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22", "tx", "ty", "tz"};
    std::array<std::size_t, 12> transform_columns{};
    for (std::size_t i = 0; i < transform_names.size(); ++i)
        transform_columns[i] = required(transform_names[i]);
    std::vector<InputPose> poses;
    std::unordered_set<std::size_t> ids;
    while (read_row(input, fields)) {
        if (fields.size() == 1 && fields.front().empty()) continue;
        if (fields.size() != header.size()) throw std::runtime_error("CSV row has a different column count");
        if (integer(fields[record_column], "record_index") != record) continue;
        InputPose input_pose;
        auto& row = input_pose.scored;
        row.pose_id = integer(fields[id_column], "pose_id");
        if (!ids.insert(row.pose_id).second) throw std::runtime_error("Duplicate pose_id in selected record");
        row.pose.fast_weighted_score = number(fields[fast_column], "Fast score");
        row.pose.clash_fraction = number(fields[clash_column], "clash_fraction");
        if (row.pose.clash_fraction < 0.0 || row.pose.clash_fraction > 1.0)
            throw std::runtime_error("Clash fraction is outside [0, 1]");
        row.pose.occupied_atoms = integer(fields[occupied_column], "occupied_atoms");
        for (std::size_t i = 0; i < transform_columns.size(); ++i) {
            const double value = number(fields[transform_columns[i]], transform_names[i]);
            if (i < 9) row.pose.transform.rotation[i] = value;
            else row.pose.transform.translation[i - 9] = value;
        }
        const auto saved = [&](const char* name) -> std::optional<double> {
            const auto column = columns.find(name);
            if (column == columns.end() || fields[column->second].empty()) return std::nullopt;
            return number(fields[column->second], name);
        };
        input_pose.saved_weighted_interaction = saved("weighted_interaction_score");
        input_pose.saved_vinardo = saved("vinardo_like_score");
        poses.push_back(input_pose);
    }
    return poses;
}

PoseRescoringResult make_pool(const std::vector<PoseAblation>& poses,
                             RemovedScore removed, std::size_t term)
{
    if (removed == RemovedScore::WeightedInteraction) weighted_interaction_families().at(term);
    if (removed == RemovedScore::Vinardo) vinardo_terms().at(term);
    PoseRescoringResult result;
    result.input_pose_count = poses.size();
    result.weighted_interaction_calculated = result.vinardo_calculated = true;
    result.poses.reserve(poses.size());
    for (const auto& pose : poses) {
        auto row = pose.baseline;
        if (removed == RemovedScore::WeightedInteraction) row.pose.weighted_interaction_score = pose.weighted_interaction_without[term];
        if (removed == RemovedScore::Vinardo) row.vinardo = vinardo_without(row.vinardo, term);
        result.poses.push_back(row);
    }
    return result;
}

} // namespace scoring_ablation
