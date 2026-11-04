# Geometric Protein–Ligand Docking

## Overview

A C++20/CUDA program for rigid protein–ligand docking in a known binding pocket.
Chemical interaction areas define protein and ligand features. Compatible
triplets provide candidate correspondences, Kabsch alignment generates rigid
poses, and clash filtering removes excessive steric overlap. Scoring and ranking
select among the surviving poses using Fast, WeightedInteraction, Vinardo-like
or Hybrid, depending on the application.
See the [algorithm description](docs/come-funziona-il-docking.md) for details.

The CUDA driver can process aggregated ligand batches and overlap CPU preparation
with one CUDA consumer. Separate experiments provide conformer ensembles,
scoring ablation and profiling without changing the rigid search protocol.

## Applications

| Executable | Purpose |
| --- | --- |
| `geometric_docking` | Single-complex CPU docking with Fast preselection, WeightedInteraction ranking and pose export. |
| `gpu_docking` | CUDA pose search, CPU/CUDA rescoring, configurable ranking and CSV/SDF export; optional batching and CPU prefetch. |
| `multi_ligand_screen` | CPU Fast screening of MOL2 records using a shared protein index; exports counts, best score and timings. |
| `multi_ligand_screen_gpu` | CUDA Fast screening with a shared index and optional joint WeightedInteraction/Vinardo-like export computed on CPU. |

## Build

Requires a C++20 compiler, Make, Eigen3, Boost and RDKit C++.
The `chem-interactions` source directory must be available at `../chem-interactions`.
The current adjacent dependency requires CMake 3.30 or newer.
CUDA builds additionally require the NVIDIA CUDA toolkit and cuSOLVER;
running CUDA applications requires a compatible NVIDIA GPU and driver.

Run from the repository root in an environment with these dependencies configured:

```sh
# CPU Release
cmake --preset cpu-release
cmake --build --preset cpu-release

# CUDA Release
cmake --preset cuda-release
cmake --build --preset cuda-release
```

Builds use `build/cpu-release/` and `build/cuda-release/`. The CUDA preset targets
A100 (architecture 80). See the [build guide](docs/build-e-test.md) for environment
setup, CMake options and other GPU architectures.

## Usage

Run from the repository root and use a new output directory for each run.
A complex directory named `<id>` contains `<id>_protein.pdb`, `<id>_pocket.pdb`
and `<id>_ligand.mol2`. Protein and pocket must share a coordinate frame;
the input ligand need not already be in the pocket.

CPU docking of one complex:

```sh
./build/cpu-release/geometric_docking /path/to/1a1e \
  --output /path/to/new-cpu-output
```

The two docking applications default to quaternion alignment. `gpu_docking`
uses optimized host queries; batching and CPU prefetch remain opt-in.

CUDA docking with Hybrid ranking, CUDA rescoring and CSV-only output:

```sh
./build/cuda-release/gpu_docking \
  /path/to/protein.pdb /path/to/pocket.pdb /path/to/ligand.mol2 \
  /path/to/new-gpu-output --limit 1 --kabsch quaternion \
  --ranking hybrid --rescoring-backend cuda --hybrid-lambda 0.3 \
  --vinardo-alpha 0.5 --fast-prefilter none --top-k 10 --no-sdf
```

This example explicitly selects rep50 and a complete rescoring pool. Hybrid
does not imply rep50 by default. See the [usage guide](docs/uso.md) for screening,
other rankings, reference inputs and optional diagnostics, and the
[batch guide](docs/gpu-batch.md) for batch/overlap configuration.

## Outputs

CPU docking writes `benchmark.csv` and `<id>/ranking.csv`, with optional PDB poses.
CUDA docking writes `records.csv`, `summary.csv` and, when enabled, `poses.csv`,
SDF poses and `timings.csv`. Export controls do not change the scoring pool.
See [output formats](docs/output.md) for export scope, pose IDs and timing limits.

RMSD and optional heavy-atom centroid distance evaluate poses against an explicit
reference and do not enter energy rankings. The conformer suite also provides an
explicit RMSD Oracle diagnostic, which requires the native reference.

## Tests and Experiments

After building, run the matching test preset in the CPU or NVIDIA environment:

```sh
ctest --preset cpu-release
ctest --preset cuda-release
```

The [test guide](tests/README.md) describes synthetic and real-data integration
checks, including GPU and molecular-input requirements. The
[experiments guide](experiments/README.md) covers diagnostics, profiling, scoring
studies and separate experimental suites, with their scope and validation status.
The [C++ conformer suite](experiments/ligand_conformer_docking/README.md) has a
separate build for ensemble docking and conformational-quality analysis.
