#ifndef OVERLAP_TIMING_HPP
#define OVERLAP_TIMING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Seconds on one steady-clock origin shared by the producer and consumer.
// Fast keeps its original search+selection span. CUDA rescoring records separate
// geometry, rescoring and whole-dock wrappers. None is isolated kernel time.
struct OverlapRecordTiming {
    std::size_t record_index = 0;
    std::string status;
    std::optional<double> query_start_s = std::nullopt, query_end_s = std::nullopt;
    std::optional<double> search_start_s = std::nullopt, search_end_s = std::nullopt;
    double consumer_wait_s = 0.0;
    std::optional<double> dock_start_s = std::nullopt, dock_end_s = std::nullopt;
    std::optional<double> rescoring_start_s = std::nullopt, rescoring_end_s = std::nullopt;
};

struct OverlapStatistics {
    double query_search_host_overlap_s = 0.0;
    std::size_t query_search_overlap_records = 0;
    double query_rescoring_host_overlap_s = 0.0;
    std::size_t query_rescoring_overlap_records = 0;
    double query_dock_host_overlap_s = 0.0;
    std::size_t query_dock_overlap_records = 0;
};

// Both stages are serial in input order. Count only intersections with another
// record; touching endpoints do not constitute overlap. No CUDA timing or wait
// is introduced here. Kernel overlap still requires a device timeline.
inline OverlapStatistics measure_host_overlap(const std::vector<OverlapRecordTiming>& records) {
    struct Interval { double begin, end; std::size_t id; };
    std::vector<Interval> queries, searches, rescoring, docks;
    const auto append = [](auto& output, const auto& begin, const auto& end, std::size_t id) {
        if (begin.has_value() != end.has_value())
            throw std::invalid_argument("Incomplete overlap timing interval");
        if (!begin) return;
        if (!std::isfinite(*begin) || !std::isfinite(*end) || *begin < 0 || *end < *begin)
            throw std::invalid_argument("Invalid overlap timing interval");
        output.push_back({*begin, *end, id});
    };
    for (const auto& row : records) {
        append(queries, row.query_start_s, row.query_end_s, row.record_index);
        append(searches, row.search_start_s, row.search_end_s, row.record_index);
        append(rescoring, row.rescoring_start_s, row.rescoring_end_s, row.record_index);
        append(docks, row.dock_start_s, row.dock_end_s, row.record_index);
        if (row.rescoring_start_s && (!row.dock_start_s || !row.search_start_s))
            throw std::invalid_argument("Rescoring span requires geometry and whole-dock spans");
        if (row.dock_start_s && row.search_start_s &&
            (*row.search_start_s < *row.dock_start_s || *row.search_end_s > *row.dock_end_s))
            throw std::invalid_argument("Geometry span is outside its dock span");
        if (row.rescoring_start_s &&
            (*row.rescoring_start_s < *row.search_end_s || *row.rescoring_end_s > *row.dock_end_s))
            throw std::invalid_argument("Rescoring span precedes geometry or exceeds dock");
    }
    const auto ordered = [](const auto& intervals) {
        for (std::size_t i = 1; i < intervals.size(); ++i)
            if (intervals[i].begin < intervals[i - 1].end)
                throw std::invalid_argument("A serial stage contains overlapping/out-of-order records");
    };
    ordered(queries);
    ordered(searches);
    ordered(rescoring);
    ordered(docks);
    OverlapStatistics result;
    const auto intersect = [&](const auto& spans, double& seconds, std::size_t& count) {
        std::size_t first = 0;
        for (const auto& query : queries) {
            while (first < spans.size() && spans[first].end <= query.begin) ++first;
            double intersection = 0.0;
            for (auto i = first; i < spans.size() && spans[i].begin < query.end; ++i) {
                const auto& span = spans[i];
                if (query.id != span.id)
                    intersection += std::max(0.0, std::min(query.end, span.end) -
                                                 std::max(query.begin, span.begin));
            }
            seconds += intersection;
            count += intersection > 0;
        }
    };
    intersect(searches, result.query_search_host_overlap_s, result.query_search_overlap_records);
    intersect(rescoring, result.query_rescoring_host_overlap_s, result.query_rescoring_overlap_records);
    intersect(docks, result.query_dock_host_overlap_s, result.query_dock_overlap_records);
    return result;
}

#endif
