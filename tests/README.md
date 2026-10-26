# Tests

CTest runs correctness checks for the host code, CUDA backend and selected
real-data integration paths. Synthetic fixtures check specific invariants;
they do not establish docking accuracy or performance on a molecular dataset.

## Running the tests

Run from the repository root after building the corresponding preset, as
described in the [main README](../README.md#build).

```sh
# CPU environment; no CUDA toolkit or GPU required.
ctest --preset cpu-release

# CUDA environment with a working NVIDIA GPU.
ctest --preset cuda-release
```

Both presets enable `BUILD_TESTING` and run tests serially. The CUDA preset
includes the host tests as well. To select or inspect registered tests:

```sh
ctest --preset cuda-release -L cuda
ctest --preset cuda-release -N
```

With Python available, the root build registers 15 CPU tests, or 25 tests
with CUDA enabled. Setting `DOCKING_TEST_COMPLEX` adds three integration
cases, for 28 tests in a CUDA build.
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
| `gpu_triplet_index_test` | Hash/flat neighbor lookup content and order; index conversion; reference/optimized queries, permutations, bin boundaries and invalid inputs. | None. |
| `rigid_alignment_test` | SVD rigid fits, proper rotations, residuals and degenerate/noncongruent triangles. | CTest selects SVD. |
| `rigid_alignment_quaternion_test` | The same alignment fixtures with quaternion Kabsch. | Same executable as `rigid_alignment_test`, with the quaternion environment selector. |
| `pose_generation_test` | Original/cached clash equivalence, screening-only behavior, shared index, clearance pruning, pose order and profiling replay. | None. |
| `pose_ranking_test` | Fast/unweighted/WeightedInteraction rankings, prefilter, ties and independence from RMSD. | None. |
| `molecule_io_test` | MOL2 text streaming, record names, EOF, line endings and input errors; not RDKit chemical parsing. | Writable system temporary directory. |
| `pose_score_ranking_test` | Score directions, rep50/Hybrid normalization before Top-K, ties and invalid/nonfinite values. | None. |
| `pose_rescoring_test` | Joint scores attached to the same pose IDs/transforms, full versus prefiltered pools, scorer selection and CSV contents. | None. |
| `gpu_docking_config_test` | CLI defaults, validation, compatible options and environment precedence; input files are not opened. | None. |
| `gpu_docking_output_test` | Top-K selection, heavy-atom RMSD/reference checks, CSV/SDF export and optional rescoring fields. | None; uses synthetic molecules and memory streams. |

The two alignment entries are separate CTest cases using one executable.

## Python checks: `../scripts/`

CMake registers these host tests only when a Python 3 interpreter is found.
They use the standard library, not Python RDKit, molecular datasets or CUDA.
Without Python, the two entries are absent rather than reported as skipped.

| CTest name | What it checks | Additional prerequisites |
| --- | --- | --- |
| `test_clearance_pruning` | Campaign-runner parsing, reductions, RMSD/status handling, failures and output protection using synthetic logs and mocked runs. | Writable temporary directory. |
| `test_gpu_rescoring_comparison` | Comparison-tool protocol, counts, pose IDs/order, tolerances and invalid exports using synthetic CSVs. | Writable temporary directory; no GPU is used. |

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

## Real-data integration: `integration/`

| CTest name | What it checks | Additional prerequisites |
| --- | --- | --- |
| `cuda_real_pipeline_test` | Real target/index preparation and scans; CPU/CUDA transforms, clash counts/decisions, Fast scores and pipeline consistency on the generated alignments. | NVIDIA GPU and a readable complex directory, registered through `DOCKING_TEST_COMPLEX`. |

The executable is built whenever CUDA and testing are enabled. It is registered
in CTest only when `DOCKING_TEST_COMPLEX` is a nonempty **CMake cache path**;
setting a shell variable with that name does not register the test.

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

The test calls `run_pipeline()` independently of `GpuDockingScreen::dock()`;
the CPU receives no GPU candidates or transforms. The GPU cutoff is set to
the CPU ligand diameter, not 24 A. Both searches use the same `RunConfig`
geometry/clash parameters with clearance pruning disabled. CTest fixes the
query method to `reference`, CPU clash to `original` and stage profiling OFF.
Targets, indexed triplets, candidates, valid alignments and survivors must
be nonempty, and their comparable counts must agree exactly.

The CPU's existing `rescore_poses()` and `rank_rescored_poses()` APIs evaluate
its complete survivor pool with Hybrid lambda 0.3, Vinardo alpha 0.5 (rep50)
and no Fast prefilter. The GPU uses its CUDA rescoring backend. This compares
the common Hybrid protocol, not the legacy Fast top-500/WeightedInteraction ranking returned
by `run_pipeline()`. WeightedInteraction uses pocket areas; Vinardo uses the full protein.
RMSD never selects or orders poses.

The GPU returns all ranked survivors. Original survivor IDs map them back to
CPU generation order before comparison; duplicates and multiplicity remain
intact. The first `min(10, survivors)` ranked IDs must match strictly, including
the production tie-break. Near-tie rank differences fail with a score-gap
diagnostic; the test does not rerank with an epsilon comparator.

| Quantity | Comparison |
| --- | --- |
| Counts, clash occupancy/fraction, chemical interaction counts and default WeightedInteraction score | Exact equality. |
| Rotation coefficients / translation components | Absolute error at most `1e-6` (translation in angstroms). |
| Transformed atom coordinate components | Absolute error at most `1e-5` angstroms. |
| Fast weighted score | Absolute error at most `1e-9`. |
| Vinardo components, total and rep50 | `abs(error) <= 1e-10 + 1e-8 * abs(CPU)`. |
| Normalized Hybrid | `abs(error) <= 1e-11 + 1e-9 * abs(CPU)`. |

All compared numerical values must be finite. These bounds reuse the existing
integration/scoring tests; independently fitted poses can still cross voxel
or interaction thresholds. Such differences fail with phase, field and pose
diagnostics rather than being hidden by wider tolerances.

The CPU API does not expose the identities of all rejected alignments, so
only their counts are compared; survivors are checked by ordinal and geometry.
This test exercises the production APIs, not CLI parsing or CSV/SDF export,
and does not establish RMSD accuracy or benchmark performance.

## Separate experimental suites

The root presets do not configure the autonomous test suites for
[scoring ablation](../experiments/scoring_ablation/README.md) or
[conformer ensembles](../experiments/conformer_ensemble/README.md).
Their own CMake files register their tests; importing the engine there does
not register the root engine suite. See those READMEs for scope and status.
