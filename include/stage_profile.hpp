#ifndef STAGE_PROFILE_HPP
#define STAGE_PROFILE_HPP

#include <cstddef>
#include <iosfwd>
#include <string_view>
#include <vector>

// Labels are borrowed; their storage must outlive the profile. Intervals may be
// nested or diagnostic replays, so rows are not necessarily additive.
struct StageTiming
{
    std::string_view phase;
    double milliseconds{};
    std::size_t elements{};
    std::string_view scope;
};

using StageProfile = std::vector<StageTiming>;

// Only the exact value "1" enables profiling. Drivers read this once at startup.
bool stage_profiling_enabled();

// Separate diagnostic CSV on stderr; scientific CSV files keep their schema.
// Scopes distinguish observed, replay and mixed device/transfer intervals.
void write_stage_profile(
    std::ostream& output,
    std::string_view backend,
    std::size_t record_index,
    const StageProfile& stages);

#endif // STAGE_PROFILE_HPP
