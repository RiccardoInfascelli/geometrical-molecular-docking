#ifndef POSE_EXPORT_HPP
#define POSE_EXPORT_HPP

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

enum class PoseExportMode { None, TopK, All };
enum class PoseExportDetail { Compact, Full };

struct PoseExportOptions {
    PoseExportMode mode = PoseExportMode::TopK;
    std::size_t top_k = 20;
    // Unset preserves the executable's original CSV schema.
    std::optional<PoseExportDetail> detail;
};

inline PoseExportMode parse_pose_export_mode(const std::string& value) {
    if (value == "none") return PoseExportMode::None;
    if (value == "topk") return PoseExportMode::TopK;
    if (value == "all") return PoseExportMode::All;
    throw std::invalid_argument("--pose-output expects none, topk or all");
}

inline PoseExportDetail parse_pose_export_detail(const std::string& value) {
    if (value == "compact") return PoseExportDetail::Compact;
    if (value == "full") return PoseExportDetail::Full;
    throw std::invalid_argument("--pose-detail expects compact or full");
}

inline std::size_t parse_pose_top_k(const std::string& value) {
    std::size_t count{};
    const auto converted = std::from_chars(value.data(), value.data() + value.size(), count);
    if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || count == 0)
        throw std::invalid_argument("--top-k expects a positive integer");
    return count;
}

inline const char* pose_export_mode_name(PoseExportMode mode) {
    return mode == PoseExportMode::None ? "none" : mode == PoseExportMode::All ? "all" : "topk";
}

inline std::size_t pose_export_count(const PoseExportOptions& options, std::size_t available) {
    if (options.mode == PoseExportMode::None) return 0;
    return options.mode == PoseExportMode::All ? available : std::min(options.top_k, available);
}

#endif
