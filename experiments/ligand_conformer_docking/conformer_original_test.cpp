#include "conformer_docking.hpp"
#include "gpu_docking.hpp"
#include "molecule_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double a, double b, double tolerance, const char* message) {
    require(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tolerance, message);
}
void counts(const ConformerDockingResult& experiment, const GpuDockingResult& normal) {
    require(experiment.matches > 0 && experiment.valid_transforms > 0 && experiment.survivors > 0,
            "Configured integration complex must exercise matching, alignment and surviving poses");
    require(experiment.triplets == normal.ligand_triplets && experiment.matches == normal.candidate_matches &&
            experiment.alignments == normal.alignments && experiment.valid_transforms == normal.valid_transforms &&
            experiment.survivors == normal.survivors, "Original-geometry pipeline counts differ");
}
void geometry(const EnsemblePose& experiment, const GpuPipelinePose& normal) {
    require(experiment.alignment_index == normal.alignment_index, "Alignment identity differs");
    for (std::size_t i = 0; i < 9; ++i)
        near(experiment.scored.pose.transform.rotation[i], normal.transform.rotation[i], 1e-6, "Rotation differs");
    for (std::size_t i = 0; i < 3; ++i)
        near(experiment.scored.pose.transform.translation[i], normal.transform.translation[i], 1e-6, "Translation differs");
    require(experiment.scored.pose.occupied_atoms == normal.clash.occupied_atoms, "Clash occupancy differs");
    near(experiment.scored.pose.clash_fraction, normal.clash.fraction, 0.0, "Clash fraction differs");
    near(experiment.scored.pose.fast_weighted_score, normal.fast_score.weighted_score, 1e-9, "Fast score differs");
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Supply one real complex directory");
        for (const auto& [key, value] : {std::pair{"DOCKING_KABSCH_METHOD", "quaternion"},
             {"DOCKING_QUERY_METHOD", "reference"}, {"DOCKING_QUERY_THREADS", "1"}, {"DOCKING_PROFILE_STAGES", "0"}})
            if (::setenv(key, value, 1)) throw std::runtime_error("Cannot configure test environment");
        for (const auto* key : {"DOCKING_AREA_CROP_MARGIN_A", "DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS"})
            if (::unsetenv(key)) throw std::runtime_error("Cannot reset test environment");
        const auto config = parse_run_config(argc, argv);
        const auto reference = load_ligand(config.ligand_path.string());
        RDKit::RWMol original(*reference);
        const PreparedGpuTarget target(config, 24.0);
        const RescoringTarget scoring(config);
        const auto experiment = dock_conformer(0, original, *reference, target, scoring, config, 500);

        GpuDockingSettings fast_settings;
        GpuDockingScreen fast_screen(config, fast_settings);
        auto fast = fast_screen.dock(original, std::max<std::size_t>(1, experiment.survivors));
        counts(experiment, fast);
        const auto fast_ranking = rank_ensemble_poses(experiment.poses, PoseScoreMethod::Fast);
        require(fast_ranking.size() == fast.poses.size(), "Fast ranked pool size differs");
        for (std::size_t i = 0; i < fast_ranking.size(); ++i)
            geometry(experiment.poses.at(fast_ranking[i].source_index), fast.poses[i]);
        std::sort(fast.poses.begin(), fast.poses.end(), [](const auto& a, const auto& b) {
            return a.alignment_index < b.alignment_index;
        });
        require(fast.poses.size() == experiment.poses.size(), "Full survivor pool size differs");
        for (std::size_t i = 0; i < fast.poses.size(); ++i) geometry(experiment.poses[i], fast.poses[i]);

        std::vector<EnsemblePose> selected;
        for (auto id : experiment.rescored_ids) selected.push_back(experiment.poses.at(id));
        for (auto method : {PoseScoreMethod::WeightedInteraction, PoseScoreMethod::Vinardo, PoseScoreMethod::Hybrid}) {
            const auto ranked = rank_ensemble_poses(selected, method);
            GpuDockingSettings settings;
            settings.ranking.method = method;
            if (method != PoseScoreMethod::WeightedInteraction) settings.ranking.vinardo_alpha = 0.5;
            settings.fast_prefilter = 500;
            GpuDockingScreen screen(config, settings);
            const auto normal = screen.dock(original, 20);
            counts(experiment, normal);
            require(normal.candidate_pool == selected.size() && normal.poses.size() == std::min<std::size_t>(20, ranked.size()),
                    "Rescoring candidate pool or Top-20 size differs");
            for (std::size_t i = 0; i < normal.poses.size(); ++i) {
                const auto& pose = selected.at(ranked[i].source_index);
                require(pose.pose_id == normal.scores[i].pose_id, "Top-20 survivor identity/order differs");
                geometry(pose, normal.poses[i]);
                if (method == PoseScoreMethod::WeightedInteraction)
                    near(ranked[i].score, normal.scores[i].weighted_interaction->weighted_score, 1e-12, "WeightedInteraction differs");
                if (method == PoseScoreMethod::Vinardo)
                    near(ranked[i].score, *normal.scores[i].vinardo_adjusted, 1e-12, "Vinardo rep50 differs");
                if (method == PoseScoreMethod::Hybrid)
                    near(ranked[i].score, *normal.scores[i].hybrid, 1e-12, "Single-conformer Hybrid differs");
            }
        }
        std::cout << "Original geometry agrees with normal CUDA docking: full Fast pool and rescored Top-20\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Original-conformer integration test failed: " << error.what() << '\n';
        return 1;
    }
}
