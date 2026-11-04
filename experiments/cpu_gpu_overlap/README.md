# CPU–GPU overlap benchmark

Compare sequential preparation with a bounded CPU producer preparing the next
ligand and its triplet queries while one CUDA consumer docks the current ligand.
The same shared receptor, CUDA buffer reuse and scientific protocol are used in
OFF and ON. No concurrent calls to the GPU screen, extra streams, GPU processes,
MPS or CPU rescoring workers are introduced.

## Protocol

The default protocol uses the first 1,000 MOL2 records, Quaternion/Horn, reference queries,
shared-only index, cutoff 24 Å, Fast ranking, CROP OFF and no SDF or reference
RMSD. Each invocation prepares its own receptor. Non-Fast rankings require CUDA
rescoring; the declared pool is all clash survivors unless a positive
`--fast-prefilter` is supplied. Engine defaults are not changed by this runner.
Supply `--complex` and `--ligands` explicitly, as below, to avoid site-specific paths.

| Ranking | Runner options |
| --- | --- |
| Fast | `--ranking fast` |
| WeightedInteraction | `--ranking weighted-interaction --backend cuda` |
| Vinardo | `--ranking vinardo --backend cuda --vinardo-alpha 1` |
| Vinardo rep50 | `--ranking vinardo-rep50 --backend cuda` |
| Hybrid rep50 | `--ranking hybrid --backend cuda --vinardo-alpha 0.5 --hybrid-lambda 0.3` |

Hybrid alone defaults to alpha 1 and lambda 0.3; it does not imply rep50.
`--fast-prefilter none` is the default. `--crop-margin 8` enables the existing
chemical-area crop for every configuration in that campaign.

Supported query-thread counts are 1, 4 and 8. Defaults remain threads 1/8 and
three repetitions; `--repeats 1` is allowed and labelled
`single_repeat_exploratory` in the metadata and aggregate CSVs. One repetition
cannot estimate dispersion or establish statistical robustness. Larger
repetition counts remain available. Configuration order rotates between repeats.

Allocate at least `max(threads)+1` physical CPU cores and an otherwise idle A100.
Eight query workers require nine cores including the consumer. One-thread ON
still uses a producer and a consumer. Small inputs may use the documented serial
query fallback; requested and actual thread counts are recorded and checked.
The runner checks CPU affinity and GPU idleness and fixes BLAS/OpenMP settings
to avoid inherited oversubscription. It never submits or transfers jobs.

## Build and preflight

Run in the configured CUDA environment, from the repository root:

```sh
cmake --preset cuda-release -B build/cuda-overlap -DENABLE_OPENMP=ON
cmake --build build/cuda-overlap --target gpu_docking --parallel 2
EXE="$PWD/build/cuda-overlap/gpu_docking"
```

Use fresh, non-existing output directories. A small full-pool smoke compares
the legacy path with explicit OFF/ON, including every pose in the declared pool:

```sh
python3 -u experiments/cpu_gpu_overlap/run_benchmark.py \
  --executable "$EXE" --complex /path/to/1a1e \
  --ligands /path/to/Commercial_higher500.mol2 \
  --output /path/to/new-smoke --smoke-only --limit 3 --threads 1 4 8
```

The smoke allows at most three records. A single record checks the API but
cannot demonstrate cross-record overlap. Repeat the smoke with the same ranking
options planned for the measured campaign. Do not proceed after a mismatch.

```sh
python3 -u experiments/cpu_gpu_overlap/run_benchmark.py \
  --executable "$EXE" --complex /path/to/1a1e \
  --ligands /path/to/Commercial_higher500.mol2 \
  --output /path/to/new-hybrid-campaign --limit 1000 --threads 1 4 8 --repeats 1 \
  --ranking hybrid --backend cuda --fast-prefilter none \
  --vinardo-alpha 0.5 --hybrid-lambda 0.3
```

Use `--repeats 3` or more for repeated measurements and run scoring campaigns
separately. `--overlap off on` is the default; subsets are allowed. ON-only
campaigns report wall/throughput but have no paired OFF speedup baseline.

## Outputs and checks

Each invocation has a separate output directory, command/environment record,
stdout/stderr and validation result. `campaign.json` records the explicit
scientific protocol, source/binary hashes, Git state, build cache, inputs and
hardware. The runner checks effective Quaternion/reference, requested threads,
scoring/backend, alpha/lambda, prefilter, cutoff/index policy, CROP, pruning and
export settings against executable output, including the legacy smoke.

`runs.csv`, `summary.csv`, `runs.json` and `status.json` retain counts, failures,
work durations, actual thread usage, online throughput and paired OFF/ON speedup.
Malformed records are retained with ID/reason. Expected exit code 1 is accepted
only for complete runs whose error rows/counts agree. Short input, missing output
or scientific mismatch stops the campaign without retries or invented rates.

The full-pool smoke compares all poses; measured runs compare Top-10 output and
complete candidate/alignment/survivor counts. IDs, multiplicities, statuses and
order remain exact. Optional numeric tolerances default to zero and apply only
to finite numerical fields. These are same-backend comparisons, not claims of
bitwise equivalence between CPU and CUDA scoring.

## Timing interpretation

`online_wall_s` is directly measured from input processing through final record
export and producer join, excluding receptor startup, timeline and summary
writes. External process wall includes startup and cleanup. Preparation, search,
rescoring and docking work intervals overlap or nest; do not add them to recreate
wall time. Performance runs disable stage profiling and NVTX tracing.

`overlap_timeline.csv` compares query work with another record's completed
search/rescoring/docking host wrappers. Fast keeps its original timeline fields;
non-Fast appends the complete docking and rescoring spans. OFF must have zero
intersection. ON without search/rescoring intersection is reported as not
demonstrated and does not receive an overlap speedup claim.

Host intersections do not prove overlap with device kernels. For that claim use
a separate short Nsight Systems trace with `--overlap-trace`, verify NVTX is
available, and inspect CPU query ranges against CUDA kernel execution. Keep
profiled runs out of throughput summaries.

## Local runner tests

```sh
python3 -B -m unittest discover -s experiments/cpu_gpu_overlap -p 'test_*.py' -v
```

These tests use mocked executable outputs. They cover all rankings, 1/4/8
threads, one or repeated measurements, strict protocol validation, errors,
output comparison and timing consistency. Use the CUDA integration test for
real OFF/ON checks and the runner to measure throughput.
