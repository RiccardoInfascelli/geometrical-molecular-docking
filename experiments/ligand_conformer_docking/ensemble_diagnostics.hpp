#ifndef CONFORMER_ENSEMBLE_DIAGNOSTICS_HPP
#define CONFORMER_ENSEMBLE_DIAGNOSTICS_HPP

#include "conformer_docking.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace conformer_diagnostics {

struct Input {
    std::size_t conformer_id{};
    const RDKit::ROMol* molecule{};
    const ConformerDockingResult* docking{};
};

struct Timings {
    double metrics_s{};
    double analysis_s{};
};

// Inputs borrow completed docking results: original ID 0 first, followed by
// generated conformers in accepted order. Every survivor must have raw scores.
// Budgets 1/5/10/20 select accepted prefixes, not fresh embedding attempts.
// Global pose IDs are zero-based offsets in the complete run sorted by
// (conformer_id, pose_id), retained unchanged in every analyzed subset.
// Only aggregate CSVs are written; scoring and input records remain unchanged.
void write_diagnostics(std::ostream& summary, std::ostream& topk,
    const std::string& complex_id, const std::vector<Input>& inputs,
    const RDKit::ROMol& reference, Timings* timings = nullptr);

} // namespace conformer_diagnostics

#endif
