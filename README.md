# Geometric Protein–Ligand Docking

## Overview

A C++20/CUDA program for rigid protein–ligand docking in a known binding pocket.
Chemical interaction areas define protein and ligand features. Compatible
triplets provide candidate correspondences, Kabsch alignment generates rigid
poses, and clash filtering removes excessive steric overlap. Scoring and ranking
select the remaining candidates through CPU or CUDA implementations.
See the [algorithm description](docs/come-funziona-il-docking.md) for details.

## Applications

| Executable | Purpose |
| --- | --- |
| `geometric_docking` | Single-complex CPU docking with Fast preselection, WeightedInteraction ranking and pose export. |
| `gpu_docking` | CUDA pose search with Fast, WeightedInteraction, Vinardo-like or Hybrid ranking and Top-K CSV/SDF export. |
| `multi_ligand_screen` | CPU Fast screening of MOL2 records using a shared protein index; exports counts, best score and timings. |
| `multi_ligand_screen_gpu` | CUDA Fast screening with a shared index and optional joint WeightedInteraction/Vinardo-like export computed on CPU. |

## Build

Requires a C++20 compiler, CMake 3.30+, Make, Eigen3, Boost and RDKit C++.
The `chem-interactions` source directory must be available at `../chem-interactions`.
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
and `<id>_ligand.mol2`. Protein and pocket coordinates must share a coordinate frame.

CPU docking of one complex:

```sh
./build/cpu-release/geometric_docking /path/to/1a1e \
  --output /path/to/new-cpu-output
```

CUDA search with WeightedInteraction ranking computed on the GPU:

```sh
./build/cuda-release/gpu_docking \
  /path/to/protein.pdb /path/to/pocket.pdb /path/to/ligand.mol2 \
  /path/to/new-gpu-output --limit 1 --ranking weighted-interaction \
  --rescoring-backend cuda --top-k 10
```

RMSD is an evaluation metric, never a ranking criterion. See the
[usage guide](docs/uso.md) for screening commands, ranking options and outputs.

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
