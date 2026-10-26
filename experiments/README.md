# Experiments

## Overview

This directory contains analysis, geometry diagnostics, profiling tools and
scientific evaluation workflows. They are separate from the four operational
applications in [apps/](../apps/) and the automated checks in [tests/](../tests/README.md).
Most tools reuse the engine; some deliberately use a different search or ranking
protocol. Their presence does not enable an optimization in the main applications.

## Experiments

The status describes the tool's scope, not a claim of dataset-wide validation.
Operational workflows are implemented; diagnostic/exploratory tools investigate
specific choices; WIP features still require build and numerical verification.

| Category | Program or suite | Purpose | Status |
| --- | --- | --- | --- |
| Geometry | [`triplet_failure_diagnostic`](geometry/triplet_failure_diagnostic_main.cpp) | Explain why ligand-area triples are rejected: source identity, distances and collinearity. Reads only the ligand; no docking. | Diagnostic |
| Geometry | [`clash_threshold_experiment`](geometry/clash_threshold_experiment_main.cpp) | Examine pose generation with relaxed clash acceptance and export poses. | Exploratory; see threshold caveat below. |
| Geometry | [`pose_target_shift_experiment`](geometry/pose_target_shift_experiment_main.cpp) | Compare target placement with nearby, chemically compatible features of the known ligand. | Diagnostic; uses a geometric oracle. |
| Geometry | [`pocket_area_experiment`](geometry/pocket_area_experiment_main.cpp) | Compare whole-protein area extraction with whole-residue crops at several pocket margins. | Exploratory; no docking. |
| Geometry | [`area_accessibility_experiment`](geometry/area_accessibility_experiment_main.cpp) | Measure free space and maximum clearance around areas, including those involved in crystal contacts. | Diagnostic; does not prune targets. |
| Geometry | [`clearance_pruning_experiment`](geometry/clearance_pruning_experiment_main.cpp) | Measure how target clearance filtering changes the index, candidates, survivors, best available RMSD and runtime. | Operational comparison; no scoring/ranking. |
| Geometry | [`pocket_openness_experiment`](geometry/pocket_openness_experiment_main.cpp) | Probe straight escape directions from the known ligand's heavy-atom centroid. | Exploratory; pocket classes are provisional. |
| Geometry | [`protein_index_cutoff_bench`](geometry/protein_index_cutoff_bench.cpp) | Measure CPU hash-index construction, record counts and host-memory estimates for one cutoff. | Diagnostic profiling; no pose search or VRAM measurement. |
| Profiling | [`offline_profile_experiment`](profiling/offline_profile_experiment_main.cpp) | Time protein/pocket loading, area extraction, grid construction and target generation separately. | Diagnostic; excludes the online docking phase. |
| Molecular analysis | [`ligand_diameter_stats`](analysis/ligand_diameter_stats.cpp) | Describe interaction-area diameters in a complex collection or sampled multi-record MOL2 file; export CSV and summary. | Operational analysis; no docking. |
| Molecular analysis | [`crystal_affinity_interactions`](analysis/crystal_affinity_interactions.cpp) | Export crystal-pose chemical interaction counts and PDBbind Kd/Ki records for affinity analysis. | Exploratory; does not evaluate predicted poses. |
| Scoring | [`vinardo_rescore_experiment`](scoring/vinardo_rescore_experiment.cpp) | Compare scores on either a historical Fast top-N pool or a declared joint Fast/WeightedInteraction/Vinardo-like pool. | Operational export; protocols differ below. |
| Scoring | [`scoring_ablation`](scoring_ablation/README.md) | Replay saved transforms on a fixed pool and measure ranking changes after removing one WeightedInteraction family or Vinardo term. | Exploratory, standalone CPU suite. |
| Serial reference | [`serial_docking`](reference/serial_main.cpp) | Retain a CPU hash-index Fast→WeightedInteraction reference with its own orchestration, Top-20 report and timings. | Operational reference; not an alias of the main pipeline. |
| CUDA screening | [`multi_ligand_screen_gpu_batch`](screening_batch/multi_ligand_screen_gpu_batch.cpp) | Compare warp/block/multi-block ligand scheduling using counts, Fast Top-1 and throughput. | Exploratory; quaternion only, no WeightedInteraction/Hybrid rescoring. |
| Conformers | [`conformer_ensemble`](conformer_ensemble/README.md) | Generate ETKDGv3/MMFF94 conformers, dock each and replay a saved pool for global Hybrid ranking. | WIP, especially the global Hybrid extension. |

## Build and Usage

The root presets `cpu-release` and `cuda-release` both enable
`BUILD_EXPERIMENTS=ON`. They build the listed standalone programs except the
two autonomous suites: `scoring_ablation` and `conformer_ensemble` have their own
CMake configurations and reuse the engine's host libraries. The batch screener
additionally requires `ENABLE_CUDA=ON` and a compatible NVIDIA GPU.

After configuring the appropriate preset as described in the
[main build instructions](../README.md#build), individual targets can be built:

```sh
cmake --build --preset cpu-release --target clearance_pruning_experiment vinardo_rescore_experiment
cmake --build --preset cuda-release --target multi_ligand_screen_gpu_batch
```

Run examples from the repository root. A complex directory named `<id>` supplies
`<id>_protein.pdb`, `<id>_pocket.pdb` and, when needed, `<id>_ligand.mol2` in the
same coordinate frame. Replace the paths below with authorized inputs and new
output paths; some tools overwrite existing files.

```sh
# Compare disabled clearance pruning with a 1.25 angstrom threshold.
./build/cpu-release/clearance_pruning_experiment /path/to/1a1e \
  --compare-clearance 1.25

# Export all clash survivors with joint scores and saved transforms.
./build/cpu-release/vinardo_rescore_experiment /path/to/1a1e \
  --joint-output /path/to/new-joint-run --interaction-region pocket --cutoff 24

# Check an experimental CUDA strategy against the ordinary CUDA pipeline.
DOCKING_KABSCH_METHOD=quaternion DOCKING_QUERY_METHOD=reference \
  ./build/cuda-release/multi_ligand_screen_gpu_batch \
  /path/to/1a1e /path/to/ligands.mol2 /path/to/new-batch-run \
  --limit 10 --cutoff 24 --strategy multi-block --batch-size 8 \
  --block-size 128 --blocks-per-ligand 8 --verify
```

Clearance comparison prints counts, RMSD and separate preparation/index/search
times to stdout. `--targets-only` inspects targets without pose generation;
without either option, the historical threshold sweep runs. Disabled pruning
differs from threshold zero, which still removes negative-clearance targets.

Joint scoring writes `poses.csv` and `protocol.csv`. It has no Fast prefilter by
default; add `--prefilter N` to restrict the pool. The separate positional mode
`vinardo_rescore_experiment COMPLEX N output.csv` retains the historical Fast
top-N experiment and Vinardo ranking. Joint export preserves generation order
and does not itself compute Hybrid. WeightedInteraction uses pocket areas by default
(`--interaction-region protein` selects all areas); Vinardo always uses the full
supported protein with its atomic cutoff.

CUDA batch output is `records.csv`, `batches.csv` and `summary.csv`. Omit `--verify`
for throughput measurements: verification adds baseline work to total wall time.
Use the local READMEs for [scoring ablation](scoring_ablation/README.md) and
[conformer ensembles](conformer_ensemble/README.md), including their separate
builds, tests and Python/RDKit requirements. Root presets do not build those suites.

## Scientific scope

Geometry diagnostics explain search behavior; timing experiments measure cost;
accuracy studies require crystal references, a declared RMSD definition and a
fixed evaluation protocol. Best available RMSD measures search coverage, not
the accuracy of a score-selected Top-1. RMSD never serves as a ranking criterion.

Do not assume identical candidate pools across tools. Serial and clearance
experiments use a ligand-diameter index; joint scoring retains a diameter
fallback above its configured cutoff; CUDA batch uses a fixed shared-only
index. Declare the pool, prefilter, receptor region, score parameters and
tie-breaking before comparing results. WeightedInteraction ablation must recompute the
maximum per area; Hybrid normalization must be recomputed on the fixed pool.

Target-shift and openness diagnostics use the known ligand and are not blind
pocket-search methods. Crop experiments do not activate cropping in the engine.
The clash experiment sets an operational threshold of **1.0**, despite its
stale **6%** banner; identify its protocol from the source parameters.

The conformer suite, particularly global Hybrid rescoring, remains **WIP: not
yet compiled and numerically validated**. Its intended normalization covers
all generated-conformer survivors before local Top-K selection, with the
crystallographic baseline kept separate. Existing source and prepared tests
do not establish a validated accuracy result.

Save the command, build/version, configuration, input identity and outputs for
each run in a new location. Preserve failures and empty pools as well as successes.
