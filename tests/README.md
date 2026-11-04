# Tests

CTest runs correctness checks for the host code, CUDA backend and selected
real-data integration paths. Synthetic fixtures check specific invariants;
they do not establish docking accuracy or performance on a molecular dataset.

## Running the tests

Run from the repository root after building the corresponding preset, as
described in the [main README](../README.md#build).

```sh
# Host checks; no CUDA toolkit or GPU required in the CPU build.
ctest --preset cpu-release -LE integration

# CUDA synthetic checks; a working NVIDIA GPU is still required.
ctest --preset cuda-release -LE integration

# Full configured CUDA suite, including real-data integration when enabled.
ctest --preset cuda-release
```

Both presets enable `BUILD_TESTING` and run tests serially. The CUDA preset
includes the host tests as well. `-LE integration` excludes real-data cases;
it does not make CUDA synthetic tests runnable without a GPU. To select only
host checks from a CUDA build or inspect registration:

```sh
ctest --preset cuda-release -L cpu
ctest --preset cuda-release -N
```

With Python available and `BUILD_EXPERIMENTS=ON`, the root build registers 21 CPU tests, or 33 tests
with CUDA enabled. Setting `DOCKING_TEST_COMPLEX` adds eleven integration
cases, for 44 tests. Disabling experiments omits the two joint diagnostic tests.
These counts describe registration, not results of an executed suite.

## Host synthetic tests: `cpu/`

These tests require the project's host libraries, but no CUDA or external
molecular files. The `gpu_` prefix in several names refers to host-side
data representations, not execution of device kernels.

| CTest name | What it checks | Additional prerequisites |
| --- | --- | --- |
| `gpu_fast_scoring_test` | Host packing of matching-area coordinates, chemical masks and Fast weights. | None; no scoring kernel is run. |
| `gpu_grid_test` | Host grid conversion, voxel occupancy, masks and dimensions. | None. |
| `gpu_geometry_test` | Conversion and ordering of ligand features, protein targets and atom coordinates. | None. |
| `gpu_triplet_index_test` | Hash/flat lookup and packing; reference/optimized query equivalence, ordering, boundaries, overflow and serial/OpenMP behavior. | OpenMP checks use `ENABLE_OPENMP=ON`; the default build checks unavailable-thread errors. |
| `gpu_ligand_prefetch_test` | Serial/prefetched molecule/query equality, bounded FIFO count/byte backpressure, ownership, cancellation, fatal-prefix drain and producer-thread OpenMP queries. | Synthetic MOL2; OpenMP checks require `ENABLE_OPENMP=ON`. |
| `gpu_batch_retry_test` | Contiguous adaptive splits preserve record order and multiplicity; singleton fresh-scratch retry, explicit terminal failure and no retry of generic CUDA errors. | Mocked batch work; no CUDA. |
| `overlap_timing_test` | Cross-record host interval intersections, sequential boundaries and invalid timing data. | None; does not establish device overlap. |
| `rigid_alignment_test` | SVD rigid fits, proper rotations, residuals and degenerate/noncongruent triangles. | CTest selects SVD. |
| `rigid_alignment_quaternion_test` | The same alignment fixtures with quaternion Kabsch. | Same executable as `rigid_alignment_test`, with the quaternion environment selector. |
| `pose_generation_test` | Original/cached clash equivalence, screening-only behavior, shared index, clearance pruning, pose order and profiling replay. | None. |
| `pose_ranking_test` | Fast/unweighted/WeightedInteraction rankings, prefilter, ties and independence from RMSD. | None. |
| `pose_output_test` | CPU none/Top-K/all export, compact/full/legacy CSVs, optional center distance, reference validation and unchanged ranking/RMSD fields. | Writable temporary directory; creates synthetic MOL2 fixtures. |
| `molecule_io_test` | MOL2 streaming; PDB whole-residue box crop, sanitization, atom order, OFF equivalence, selector validation and unchanged full-protein coordinates/connectivity. | Writable temporary directory. |
| `pose_score_ranking_test` | Score directions, rep50/Hybrid normalization before Top-K, ties and invalid/nonfinite values. | None. |
| `pose_rescoring_test` | Joint scores attached to the same pose IDs/transforms, full versus prefiltered pools, scorer selection and CSV contents. | None. |
| `joint_diagnostics_test` | Heavy-atom metrics, Full/Fast Top-500 pools, independent Hybrid normalization, cumulative Top-K, IDs, ties and empty/invalid inputs. | `BUILD_EXPERIMENTS=ON`; synthetic molecules. |
| `gpu_docking_config_test` | CLI defaults, overlap/batch and center-reference guards, compatible options and environment precedence; input files are not opened. | None. |
| `gpu_docking_output_test` | Top-K, RMSD/reference checks, heavy-atom centroid diagnostics, CSV/SDF policies, cached pool metrics and optional scores. | None; uses synthetic molecules and memory streams. |

The two alignment entries are separate CTest cases using one executable.

## Python checks

CMake registers these host tests only when a Python 3 interpreter is found.
They use the standard library, not Python RDKit, molecular datasets or CUDA.
Without Python, these entries are absent rather than reported as skipped.

| CTest name | What it checks | Additional prerequisites |
| --- | --- | --- |
| `test_clearance_pruning` | Campaign-runner parsing, reductions, RMSD/status handling, failures and output protection using synthetic logs and mocked runs. | Writable temporary directory. |
| `test_gpu_rescoring_comparison` | Comparison-tool protocol, counts, pose IDs/order, tolerances and invalid exports using synthetic CSVs. | Writable temporary directory; no GPU is used. |
| `joint_diagnostics_cli_test` | Joint diagnostic option guards, failure status/CSV escaping, empty-pool output and protection of existing output. | `BUILD_EXPERIMENTS=ON`, built `vinardo_rescore_experiment` and a writable temporary directory. |

## CUDA synthetic tests: `cuda/`

All entries require `ENABLE_CUDA=ON`, the CUDA runtime and a compatible NVIDIA
GPU. The supplied CUDA preset targets A100 (architecture 80). Missing GPU or
runtime support causes failure; these tests do not silently skip device work.
No real molecular input files are required.

| CTest name | What it checks | Additional prerequisites |
| --- | --- | --- |
| `cuda_smoke_test` | Device allocation, a minimal kernel launch and the downloaded result. | None beyond the CUDA requirements above. |
| `cuda_triplet_lookup_test` | Exact/neighbor lookup, counts, scans and triplet/alignment expansion; host/device wrapper agreement and order; empty/invalid/overflow cases. | None. |
| `cuda_geometry_upload_test` | Point upload/download, array counts and cleanup/reset. | None. |
| `cuda_kabsch_test` | SVD fits against a host-SVD reference, proper rotations, residuals, degeneracies and invalid inputs. | CTest selects SVD. |
| `cuda_kabsch_quaternion_test` | The same fixtures using quaternion CUDA Kabsch; degenerate solutions are compared by fit quality. | Same executable as `cuda_kabsch_test`, with the quaternion selector. |
| `cuda_clash_test` | Transformed atom occupancy and clash fractions, including atoms outside the grid. | None. |
| `cuda_fast_score_test` | Weighted chemical compatibility and counters for free, occupied and out-of-grid areas. | None. |
| `cuda_pose_pipeline_test` | Kabsch, clash, survivor order/identity and Fast; host/device results, buffer ownership, profiling invariance and error cleanup. | None. |
| `cuda_pose_rescoring_test` | CPU/CUDA WeightedInteraction, Vinardo and Hybrid agreement, rep50, full/prefiltered pools, Top-K and invalid scores. | Synthetic molecular fixtures; numerical tolerances are defined in the source. |
| `cuda_screening_batch_test` | Warp/block/multi-block screening versus the original pipeline: counts, Top-1, ties, heterogeneous inputs and resource bounds. | Selects quaternion internally. |
| `cuda_ligand_batch_test` | Segmented clash/Fast versus the original kernels, heterogeneous and empty segments, stable compaction IDs, malformed ranges and cleanup. | Synthetic aggregate buffers; distinct from the older screening experiment. |
| `cuda_ligand_batch_rescoring_test` | Packed heterogeneous WI/Vinardo inputs versus the serial scorers, components, local pose offsets, independent rankings/Hybrid, invalid ranges and empty inputs. | No molecular files. |

## Real-data integration: `integration/`

| CTest name | What it checks | Additional prerequisites |
| --- | --- | --- |
| `cuda_real_pipeline_test` | Real target/index preparation and scans; CPU/CUDA transforms, clash counts/decisions, Fast scores and pipeline consistency on the generated alignments. | NVIDIA GPU and a readable complex directory, registered through `DOCKING_TEST_COMPLEX`. |
| `gpu_docking_library_test` | Real CLI with valid/invalid/valid MOL2 records: CSV-only export, recovery, fail-fast behavior and unchanged counts/poses after an error. | Same real complex and GPU; Python 3. Uses temporary inputs/outputs and quaternion Kabsch. |
| `gpu_docking_overlap_test` | OFF/ON equivalence with quaternion and Fast or CUDA WeightedInteraction/Vinardo/rep50/Hybrid: full-pool identities, scores, ordering/Top-K, timing scopes, record recovery and empty input. | Same real complex and GPU; Python 3. Separate optional CLI test, bounded by a 600-second CTest timeout. |
| `gpu_docking_batch_test` | Serial versus aggregated Fast for batch sizes 1/2/4/8: complete survivor pools, local alignment IDs, transforms, scores, Top-K, record errors, partial batches and an empty index. | Same real complex and GPU; Python 3. Requires actual candidates and survivors; 900-second timeout. |
| `gpu_docking_batch_weighted_interaction_test`, `gpu_docking_batch_vinardo_test`, `gpu_docking_batch_vinardo_rep50_test`, `gpu_docking_batch_hybrid_test` | Serial versus batch CUDA rescoring, sizes 1/2/4/8, prefilter none/7, full selected pools and Top-10, score components/IDs/transforms, per-ligand Hybrid normalization and error paths. | Same real complex, GPU and Python; separate cases with 1800-second timeouts. |
| `gpu_docking_batch_overlap_test` | Standard Vinardo serial versus batches 1/4/8/16 with OFF/ON CPU prefetch, Top-10/components/IDs, adaptive splits, queue/timeline bounds, optional center export and error shutdown. | GPU, Python and a complex such as `1a1e` with enough candidates to exercise a reduced MiB budget; 1800-second timeout. |

The C++ integration executables are built with CUDA and testing enabled, but
real-data cases are registered only when `DOCKING_TEST_COMPLEX` is a nonempty
**CMake cache path**. Python CLI cases additionally require Python 3. A shell
variable with that name does not register these tests.

For an already built CUDA preset, register an authorized complex and run it:

```sh
cmake --preset cuda-release -DDOCKING_TEST_COMPLEX=/path/to/1a1e
DOCKING_KABSCH_METHOD=svd DOCKING_QUERY_METHOD=reference \
  ctest --preset cuda-release -R '^cuda_real_pipeline_test$'
```

The directory basename determines the input names: `<id>_protein.pdb`,
`<id>_pocket.pdb` and `<id>_ligand.mol2`. Use matching structures in the same
coordinate frame. The test inherits environment selectors; the example fixes
SVD and reference queries. Select `quaternion` to check that alignment method.
To unregister the real-data cases, reconfigure with `-DDOCKING_TEST_COMPLEX=`.

This test uses a protein-triplet cutoff equal to the ligand diameter, not the
screeners' fixed shared cutoff. It compares numerical results within source
tolerances; it does not invoke the `gpu_docking` CLI, check real-data Hybrid
ranking or measure RMSD accuracy. Inspect its printed candidate/survivor
counts: the test does not explicitly require them to be nonzero.

### Independent CPU/CUDA docking

| CTest name | What it checks | Additional prerequisites |
| --- | --- | --- |
| `cpu_cuda_docking_test` | Independent production CPU/CUDA searches, every survivor's geometry/clash/Fast, full-pool WeightedInteraction/Vinardo/Hybrid and the Hybrid Top-10. | Real complex and NVIDIA GPU; CTest selects SVD. |
| `cpu_cuda_docking_quaternion_test` | The same protocol using quaternion Kabsch. | Same executable as `cpu_cuda_docking_test`, in a separate process. |

These cases use the same `DOCKING_TEST_COMPLEX` registration condition.
After configuring that path and building the CUDA preset, run:

```sh
ctest --preset cuda-release -R '^cpu_cuda_docking(_quaternion)?_test$'
```

The CPU calls `run_pipeline()` independently of `GpuDockingScreen::dock()`;
it receives no GPU candidates or transforms. Both use the ligand-diameter
cutoff, matching geometry/clash parameters and no clearance pruning. CTest
fixes reference queries, original CPU clash and stage profiling OFF. The test
requires nonempty candidates/alignments/survivors and exact comparable counts.

CPU reference APIs and CUDA rescoring evaluate every survivor, without Fast
prefilter, using Hybrid lambda 0.3 and Vinardo alpha 0.5. This is separate from
the CPU application's default Fast top-500/WeightedInteraction ranking.
WeightedInteraction uses pocket areas; Vinardo uses the complete protein.
Original survivor IDs preserve duplicate occurrences and generation order.
Top-10 identities and tie-breaking must match; near-tie differences fail
instead of being hidden by an epsilon comparator. RMSD never selects poses.

| Quantity | Comparison |
| --- | --- |
| Counts, clash occupancy/fraction, chemical interaction counts and default WeightedInteraction score | Exact equality. |
| Rotation coefficients / translation components | Absolute error at most `1e-6` (translation in angstroms). |
| Transformed atom coordinate components | Absolute error at most `1e-5` angstroms. |
| Fast weighted score | Absolute error at most `1e-9`. |
| Vinardo components, total and rep50 | `abs(error) <= 1e-10 + 1e-8 * abs(CPU)`. |
| Normalized Hybrid | `abs(error) <= 1e-11 + 1e-9 * abs(CPU)`. |

Values must be finite; threshold crossings fail with phase/pose diagnostics.
Rejected alignment identities are not exposed by the CPU API, so only their
counts are compared. CLI and export behavior are covered separately.

## Separate experimental suites

The root presets do not register autonomous experimental suites. Each has
its own build directory and CTest registration; importing the engine there
does not register unbuilt root test executables.

### C++ conformer docking and quality

[ligand_conformer_docking](../experiments/ligand_conformer_docking/README.md)
registers eight host cases with Python available, or ten with CUDA and a real
`DOCKING_TEST_COMPLEX`. These are registration counts, independent of the root
suite. Configure and build all targets in that suite before running:

```sh
ctest --test-dir build/conformer-host -LE integration --output-on-failure
ctest --test-dir build/conformer-cuda -E '^conformer_(original|execution)$' --output-on-failure
ctest --test-dir build/conformer-cuda --output-on-failure
```

| CTest name | What it checks | Prerequisites |
| --- | --- | --- |
| `conformer_quality_smoke` | Quaternion/Horn aligned RMSD, proper rotations, invalid input, coordinate-free generation input and a tiny seeded ETKDG sequence. | Host RDKit C++; no molecular dataset. |
| `conformer_generation` | ETKDG repeatability, atom/stereo identity, diversity selection and MMFF behavior. | Host RDKit C++; synthetic molecules. |
| `ensemble_ranking` | Score directions, one global Hybrid normalization, stable local/global ranks, duplicates, evaluation and separate RMSD Oracle ordering. | Host libraries. |
| `ensemble_diagnostics` | Accepted conformer prefixes, original included/excluded, global Fast Top-500, independent Hybrid normalization, Top-K metrics, stable IDs and missing/empty inputs. | Host libraries; synthetic completed docking results. |
| `rmsd_oracle` | Direct heavy-atom RMSD without fitting or symmetry correction, strict reference mapping, baseline exclusion and the selected-pose/Oracle gap. | Synthetic molecules; no GPU. |
| `conformer_single_run_workflow` | Runner arguments, independent output directories, failure handling and protocol/source recording. | Python 3; subprocesses are mocked. |
| `conformer_quality_runner` | Prefix-budget CSVs, status/stereo handling, parameter-compatible resume and interrupted runs; includes a tiny compiled-runner smoke check. | Python 3 and the built `conformer_quality` executable; 30-second timeout. |
| `conformer_export` | Real CSV/SDF writers and CLI parsing on synthetic poses, output controls, center-distance cache and Oracle/shared-timing fields. | Writable temporary directory; no CUDA search. |
| `conformer_original` | Original geometry versus normal rigid docking: full Fast pool and CPU-rescored Top-20 with Fast top-500. | CUDA and real complex; exclude by name for host-only runs. |
| `conformer_execution` | Complete survivor pools across reference/optimized queries, CPU/CUDA rescoring, batch and overlap; global Hybrid, Oracle, center distance and empty/error cases. | CUDA and real complex; labeled `integration`, 1800-second timeout. |

The two real-data cases select quaternion Kabsch internally and remain distinct:
`conformer_original` checks the original protocol; `conformer_execution`
checks shared execution without a Fast prefilter. The standalone CUDA command
above excludes both by name because `conformer_original` has no integration
label. Neither is a benchmark.

[Scoring ablation](../experiments/scoring_ablation/README.md) and the separate
[Python conformer workflow](../experiments/conformer_ensemble/README.md) have
their own tests and prerequisites. The latter includes an explicit Python
RDKit availability check. See their READMEs for commands and validation scope.
