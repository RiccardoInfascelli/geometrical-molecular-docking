#include "overlap_timing.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F operation) {
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Invalid interval was accepted");
}
}

int main() {
    try {
        const std::vector<OverlapRecordTiming> serial{
            {1, "ok", 0.0, 1.0, 1.0, 3.0}, {2, "ok", 3.0, 4.0, 4.0, 6.0}};
        require(measure_host_overlap(serial).query_search_host_overlap_s == 0,
                "Touching sequential intervals were reported as overlap");
        auto pipelined = serial;
        pipelined[1].query_start_s = 1.5;
        pipelined[1].query_end_s = 2.0;
        const auto result = measure_host_overlap(pipelined);
        require(result.query_search_host_overlap_s == 0.5 && result.query_search_overlap_records == 1,
                "Cross-record overlap duration/count differs");
        require(measure_host_overlap({{1, "error"}}).query_search_overlap_records == 0,
                "Failed/no-query records invented overlap");
        // Concurrent threads can still run sequential phases: time intersections,
        // not thread count, determine whether useful host overlap was observed.
        require(measure_host_overlap({{1, "ok", 0.0, 2.0, 0.0, 2.0}})
                    .query_search_host_overlap_s == 0,
                "Same-record work was counted as cross-record overlap");
        rejects([] { measure_host_overlap({{1, "ok", 2.0, 1.0}}); });
        rejects([] { measure_host_overlap({{1, "ok", 0.0, std::nullopt}}); });
        rejects([] { measure_host_overlap({{1, "ok", 0.0, std::numeric_limits<double>::infinity()}}); });
        rejects([] { measure_host_overlap({{1, "ok", 0.0, 2.0}, {2, "ok", 1.0, 3.0}}); });
        // A query may overlap only CUDA rescoring, with no geometry overlap.
        // Whole-dock includes both stages and must not be added to its children.
        auto rescoring_only = std::vector<OverlapRecordTiming>{
            {1, "ok", 0.0, 1.0, 1.0, 2.0}, {2, "ok", 2.5, 3.0, 4.0, 5.0}};
        rescoring_only[0].dock_start_s = 1.0;
        rescoring_only[0].dock_end_s = 4.0;
        rescoring_only[0].rescoring_start_s = 2.0;
        rescoring_only[0].rescoring_end_s = 4.0;
        rescoring_only[1].dock_start_s = 4.0;
        rescoring_only[1].dock_end_s = 6.0;
        rescoring_only[1].rescoring_start_s = 5.0;
        rescoring_only[1].rescoring_end_s = 6.0;
        const auto extended = measure_host_overlap(rescoring_only);
        require(extended.query_search_host_overlap_s == 0 &&
                extended.query_rescoring_host_overlap_s == 0.5 &&
                extended.query_dock_host_overlap_s == 0.5 &&
                extended.query_rescoring_overlap_records == 1 && extended.query_dock_overlap_records == 1,
                "Rescoring-only overlap was lost or attributed to geometry");
        auto invalid = rescoring_only;
        invalid[0].rescoring_start_s = 1.5;
        rejects([&] { measure_host_overlap(invalid); });
        invalid = rescoring_only;
        invalid[0].dock_end_s = 3.5;
        rejects([&] { measure_host_overlap(invalid); });
        invalid = rescoring_only;
        invalid[0].rescoring_end_s.reset();
        rejects([&] { measure_host_overlap(invalid); });
        invalid = rescoring_only;
        invalid[0].dock_start_s.reset();
        invalid[0].dock_end_s.reset();
        rejects([&] { measure_host_overlap(invalid); });
        OverlapRecordTiming failed{1, "error"};
        failed.dock_start_s = 1.0;
        failed.dock_end_s = 2.0;
        require(measure_host_overlap({failed}).query_dock_host_overlap_s == 0,
                "A failed record invented substage intervals");
        std::cout << "Overlap timing tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
