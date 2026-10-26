# OpenMP query benchmark on A100

This standard-library Python runner invokes the existing `gpu_docking` executable;
it does not split, filter, reorder or rewrite scientific inputs. Build the executable
with CUDA and `-DENABLE_OPENMP=ON` first. OpenMP is optional and defaults to OFF.
`DOCKING_QUERY_THREADS=1` is the serial query baseline; positive values request the
OpenMP query loop, which requires at least 2048 ligand triplets in total and
1024 per requested thread; smaller workloads stay serial. Protein
and ligand preparation remain serial. Each process handles its records in order;
there is no preparation/search overlap.

```sh
cmake --preset cuda-release -B build/cuda-openmp -DENABLE_OPENMP=ON
cmake --build build/cuda-openmp \
  --target gpu_docking multi_ligand_screen_gpu query_preparation_bench --parallel 2
```

The CUDA preset selects A100 architecture 80; retain the working site's
dependency/toolchain settings. The runner stops after the first baseline if the executable reports that
OpenMP is unavailable in a campaign requesting more than one thread.

Run this on the allocated A100 node with enough allocated CPUs for the largest
thread count. Supply protein and pocket paths that share a coordinate frame.
Initial ligand coordinates need not already place the ligand in the pocket:

```sh
python3 experiments/openmp_queries/run_benchmark.py \
  --protein /absolute/path/protein.pdb \
  --pocket /absolute/path/pocket.pdb \
  --executable /absolute/path/build/cuda-openmp/gpu_docking \
  --output /absolute/path/new-openmp-campaign
```

The output directory must be new. The default library is
`/work/rinfascelli/Commercial_higher500.mol2`, and `--limit 1000` selects its first
1000 original records. There are three repeats for threads 1, 2, 4 and 8, for:

| Ranking | Rescoring backend | Fixed scoring parameters |
| --- | --- | --- |
| Fast | One run configuration; backend is irrelevant | No rescoring prefilter |
| WeightedInteraction | CPU and CUDA separately | Fast prefilter 500; pocket interactions |
| vinardo-rep50 | CPU and CUDA separately | Fast prefilter 500; repulsion alpha 0.5 |
| Hybrid | CPU and CUDA separately | Fast prefilter 500; lambda 0.3; alpha 0.5; pocket interactions |

This is 84 executable invocations by default. Every invocation uses cutoff 24 Å,
quaternion Kabsch, top-k 10, `--timings`, diagnostics off and stage profiling off.
Every invocation uses `--no-sdf` for a CSV-only benchmark: no SDF files are
written, while scientific CSV values and ordering remain unchanged.
`--top-k K` changes the exported pose limit for the whole campaign. The legacy
`DOCKING_STAGE_TIMINGS` and `DOCKING_KABSCH_TIMINGS` switches are removed from the
executed environment because even their value `0` enables extra synchronization.
Search always uses CUDA; `--backend` controls rescoring only. A single query
method is held fixed for the campaign (`--query-method reference`, the default,
or `optimized`). `--crop-margin off` removes an inherited crop setting; a positive
value sets the same chemical-area crop for every invocation. No ranking, cutoff,
crop or query-method parameter changes between thread counts within a group.

Subsets and smaller synthetic/development campaigns can be selected explicitly:

```sh
python3 experiments/openmp_queries/run_benchmark.py \
  --protein /absolute/path/protein.pdb --pocket /absolute/path/pocket.pdb \
  --executable /absolute/path/build/cuda-openmp/gpu_docking --output /absolute/path/another-new-campaign \
  --ligands /absolute/path/library.mol2 --limit 1000 --threads 1 2 4 \
  --repeats 2 --ranking weighted-interaction hybrid --backend cpu \
  --query-method optimized --crop-margin off
```

Thread count 1 is always included once as a configuration, even when omitted
from `--threads`. Its first repeat is the comparison baseline for every repeat
and thread count with the same ranking and rescoring backend. A separate
"parallel threads 1" configuration would be identical and is not added.

The runner passes `--continue-on-error` to preserve the first 1000 original
records, including malformed records. This opt-in executable mode writes
`status=error` and the error message in `records.csv`, retains original indices,
and omits failed records from `timings.csv`. It reports
`status=complete_with_errors` and exits 1 when record errors occurred. The runner
accepts that outcome only when the summary and record counts agree and all 1000
requested records were processed. The executable's default behavior without
this flag still stops at the first error. Short input, missing summaries,
unexpected exit codes or startup/runtime failure stop the campaign and preserve
partial outputs; they never produce completed-campaign throughput or speedups.
Error counts may differ across rankings or rescoring backends because they can
reject different unsupported ligand chemistry. Counts and errors are compared
only within the same ranking/backend group.

`campaign.json` records options, executable and runner SHA-256, input path/size/
mtime, Git commit/branch/status/diff, selected environment variables and optional
read-only `nvidia-smi` hardware information. Input file contents are not copied or
hashed. `command.json`, `stdout.log`, `stderr.log`, the executable's full output
directory and `result.json` are retained for every run. Environment variables
with `DOCKING_`/`OMP_` prefixes, CUDA visibility and selected Slurm fields are
recorded; unrelated environment variables are omitted.

`runs.json` contains every run, while `summary.json` groups repeats by ranking,
backend and thread count. External process wall time includes process startup
and teardown. Engine `startup_s` and `wall_before_summary_s` are reported
separately. `online_wall_s` subtracts startup from engine wall time, and
`successful_online_records_per_s` divides successful records by that interval
only for completed, valid runs with a positive interval. This online interval
includes input parsing, I/O and pose export; it is not pure GPU time. Every
timing column has its own sum, mean and nonblank count;
rescoring subintervals can overlap other measured intervals and must not be
added together to invent end-to-end wall time. `query_preparation_s` and
`gpu_search_s` are reported separately, alongside their encompassing
`search_completed_s`: query preparation plus GPU search replaces that parent
interval when decomposing search. `gpu_search_s` includes host packing,
allocation, copies, completed device waits and cleanup; it is not isolated kernel
time. CPU `hybrid_normalization_s` is already included in `top_k_selection_s`.
The summarizer never adds different timing columns together. Actual query
thread counts are metadata rather than timing
intervals. Summary groups show the counts actually used, and clearly mark an
all-serial fallback caused by small queries or an inherited `OMP_THREAD_LIMIT`
(which is recorded). Throughput distinguishes
successfully processed records (including `no_poses`/`no_triplets`) from completed
records (including errors), and separately reports records with exported poses.
No rate calls malformed records successful.

Correctness compares every field of `records.csv` and `poses.csv`, including
error rows, row/column order, all exported Top-K ranks, alignment identifiers,
scores and rigid transforms. The benchmark comparison covers exported Top-K
poses; full host query contents and ordering are checked by the C++ unit tests.
For a separate full-survivor check on one ligand, use `--ranking fast --limit 1`
and a sufficiently large `--top-k`; verify that `exported_poses` equals `survivors`
in every successful record. Keep this export-heavy check outside throughput runs.
The default is exact CSV cell equality within each
backend. Optional `--absolute-tolerance` and `--relative-tolerance` allow finite
numeric score/transform differences; identities, statuses, error messages and
integer counts remain exact. CPU and CUDA rescoring are not compared with each
other. Speedups are suppressed for incomplete or mismatching groups; the
runner returns nonzero if any comparison fails. Error-containing, fully processed
campaigns with matching results return zero.

The separate synthetic host benchmark needs no molecular data or CUDA execution.
With `BUILD_EXPERIMENTS=ON`, build and invoke it explicitly:

```sh
cmake --build build/cuda-openmp --target query_preparation_bench --parallel 2
DOCKING_QUERY_METHOD=reference DOCKING_QUERY_THREADS=4 OMP_DYNAMIC=FALSE \
  build/cuda-openmp/query_preparation_bench 16384 20
```

Its positional arguments are triplet count and repetitions; environment selectors
choose the query method and requested threads. It checks the full generated host
query/order arrays against the serial reference and reports actual threads,
checksum and host preparation intervals. Hold the method and triplet count fixed
when comparing thread counts. This synthetic result does not replace the A100
campaign's end-to-end throughput.

Run the small synthetic tests locally; docking and hardware subprocesses are
mocked and no library data is opened:

```sh
python3 -m unittest discover -s experiments/openmp_queries -p 'test_*.py' -v
```
