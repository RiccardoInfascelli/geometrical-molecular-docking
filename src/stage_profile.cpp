#include "stage_profile.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <ostream>

bool stage_profiling_enabled()
{
    const char* value = std::getenv("DOCKING_PROFILE_STAGES");
    return value && std::strcmp(value, "1") == 0;
}

void write_stage_profile(
    std::ostream& output,
    std::string_view backend,
    std::size_t record_index,
    const StageProfile& stages)
{
    const auto precision = output.precision();
    const auto flags = output.flags();
    output << "PROFILE_STAGE,backend,record_index,phase,milliseconds,elements,scope\n"
           << std::fixed << std::setprecision(6);
    for (const auto& stage : stages) {
        output << "PROFILE_STAGE," << backend << ',' << record_index << ','
               << stage.phase << ',' << stage.milliseconds << ','
               << stage.elements << ',' << stage.scope << '\n';
    }
    output.flags(flags);
    output.precision(precision);
}
