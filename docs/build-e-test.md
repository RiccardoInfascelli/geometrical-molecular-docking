# Build and Configuration

## Requirements

- A C++20 compiler, CMake and Make (the presets use `Unix Makefiles`).
- Eigen3, Boost `program_options` and RDKit C++ libraries. Boost and RDKit
  must provide discoverable CMake package configurations.
- The `chem-interactions` source directory alongside this repository:
  `../chem-interactions`. The root build adds it as a subdirectory.
- For CUDA: the NVIDIA CUDA toolkit, including cuSOLVER, and a compatible
  host compiler. Running CUDA applications and tests also requires a supported
  NVIDIA GPU and driver. CUDA sources use C++17; host sources use C++20.

The project and preset format require CMake 3.20 or newer, but the current
`chem-interactions` checkout explicitly sets `CMP0167 NEW`, making **CMake
3.30 or newer** necessary. The build files do not specify minimum versions
for Eigen3, Boost, RDKit or the CUDA toolkit.

Python 3 is optional for the main build and enables additional Python tests.
Python RDKit is not required by the engine; it is needed for conformer generation
in the separate experimental suite.

## CMake organization and options

The root build defines shared host libraries for preparation, search, scoring
and ranking. Enabling CUDA adds the device backend, GPU search and rescoring
libraries. Applications, tests and root-project experiments link these libraries.

| Option | Default | Purpose |
| --- | --- | --- |
| `ENABLE_CUDA` | `OFF` | Build the CUDA backend, GPU applications and applicable tests/experiments. |
| `BUILD_TESTING` | `ON` | Build test executables and enable CTest registration. |
| `BUILD_EXPERIMENTS` | `ON` | Build the root project's diagnostic, reference and experimental tools. |
| `DOCKING_TEST_COMPLEX` | Empty | Register real-data integration cases when CUDA and testing are enabled; set to a complex directory. |
| `TARGET_CPU_ARCHITECTURE` | `generic` | Dependency option controlling architecture flags on applicable Release C++ targets; both presets retain `generic`. |
| `CMAKE_CUDA_ARCHITECTURES` | `80` in the CUDA preset | Select the CUDA target architecture; 80 targets NVIDIA A100. |

The autonomous `scoring_ablation` and `conformer_ensemble` suites have separate
CMake builds. They are not included by `BUILD_EXPERIMENTS`; see the
[experiments guide](../experiments/README.md).

## CPU Release

Run from the repository root in an environment with the dependencies configured:

```sh
cmake --preset cpu-release
cmake --build --preset cpu-release
```

This configures `Release` in `build/cpu-release/`, with CUDA disabled and tests
and root-project experiments enabled. It builds the host applications and
libraries without requiring the CUDA toolkit or an NVIDIA GPU.

## CUDA Release

Load the CUDA/compiler and molecular-library environment before configuring:

```sh
cmake --preset cuda-release
cmake --build --preset cuda-release
```

This uses `build/cuda-release/` and inherits the CPU preset's Release, testing,
experiment and generic CPU settings, while enabling CUDA and architecture 80.
For another GPU, override the architecture during configuration:

```sh
cmake --preset cuda-release -DCMAKE_CUDA_ARCHITECTURES=<architecture>
```

Replace `<architecture>` with the appropriate compute-capability value.
Both build presets use one job to limit compilation memory; this does not
restrict runtime CPU/CUDA parallelism. Executables are placed in the selected
build directory. A specific target can be built with
`cmake --build --preset <preset> --target <target>`.

## Docker and cluster environments

Run the same commands inside the configured CPU container or cluster shell;
no particular container name, mount point or workspace path is required.
The sibling `chem-interactions` directory must be visible in that environment.
If package discovery needs an explicit prefix, pass
`-DCMAKE_PREFIX_PATH=/path/to/dependency-prefix` to the configure command.

On a cluster, load the site's compiler, CUDA and RDKit environment first, and
use a GPU allocation when running device tests. Configure build trees separately
in each environment: generated CMake files contain toolchain and source paths
and should not be copied between the host, container and cluster. CMake caches
retain explicit overrides; rerunning a preset does not clear unrelated options.

## Tests

After building, run the matching suite:

```sh
ctest --preset cpu-release
ctest --preset cuda-release
```

The CUDA suite includes host tests and requires an NVIDIA GPU for device checks.
Real-data integration tests are opt-in through the CMake cache variable
`DOCKING_TEST_COMPLEX`; a shell variable alone does not register them.
See the [test guide](../tests/README.md) for prerequisites, registration,
selection commands and numerical comparison protocols.
