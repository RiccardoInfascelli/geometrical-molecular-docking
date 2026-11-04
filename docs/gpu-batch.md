# CUDA multi-ligand batching

`gpu_docking --gpu-batch on` aggregates CUDA work across multiple MOL2 records.
It requires quaternion Kabsch, the `gpu_docking` default, and is disabled by default. Fast is supported
directly; WeightedInteraction, Vinardo, rep50 and Hybrid require
`--rescoring-backend cuda`.
It is a mode of the operational `gpu_docking` application, separate from the
experimental warp/block screeners.

## Execution

The receptor, shared triplet index and grid are prepared and uploaded once.
CPU preparation is synchronous by default. Add `--cpu-gpu-overlap on` to prepare
future records while one CUDA consumer processes the current batch. Explicit
`--cpu-gpu-overlap off` uses the same preparation synchronously.
Queries, matching points, candidates and alignments share aggregate arrays with
per-ligand offsets. The ordinary lookup and quaternion alignment kernels process
those arrays; segmented clash and Fast scoring select each ligand's atoms and
features. Stable compaction preserves accepted poses and their original IDs.
The caller receives independent survivor pools with local alignment IDs.
No candidate is truncated, deduplicated or silently processed through a serial
fallback. This is batching by phase, without concurrent CUDA streams.

With overlap ON, the producer owns a FIFO of at most two batches (`2 * batch-size`
records), with a 256 MiB estimated queued-payload limit and backpressure.
One in-flight preparation is outside that byte limit; parsing temporaries and
RDKit internals are excluded, so it is not an RSS cap. An oversized individual
packet fails explicitly. Ownership transfers only after complete preparation.
No CUDA calls run in the producer. `DOCKING_QUERY_THREADS=8` uses the existing
OpenMP query code when built with `ENABLE_OPENMP=ON`; small inputs may use fewer
threads. Chemical preparation remains serial within that producer.

For rescoring, the aggregate survivor pool remains on device. Chemistry
preparation and Fast prefiltering remain separate per ligand. Packed chemical
features let one WeightedInteraction and/or Vinardo launch score the selected
poses across the batch, using the same formulas as serial docking.

Ranking and Hybrid normalization use separate calls to the existing CUDA ranker
for each ligand. Hybrid therefore normalizes that ligand's selected population,
before Top-K, with the same ranking rules and tie-breaks as serial docking.
Ligands are never normalized together. `--fast-prefilter none|N` retains its
per-ligand meaning. Rep50 and configurable alpha/lambda are applied by the same
ranking code. Fast selection remains on the host, as in ordinary Fast docking.

```sh
./build/cuda-release/gpu_docking \
  /path/to/1a1e/1a1e_protein.pdb /path/to/1a1e/1a1e_pocket.pdb \
  /path/to/ligands.mol2 /path/to/new-batch-output \
  --limit 100 --cutoff 24 --kabsch quaternion --query-method reference \
  --ranking vinardo --rescoring-backend cuda --fast-prefilter none --vinardo-alpha 1 \
  --gpu-batch on --batch-size 16 --batch-memory-mib 4096 --cpu-gpu-overlap on \
  --top-k 10 --pose-output topk --pose-detail full --no-sdf --timings \
  --continue-on-error
```

Batch size is 1–64 input records (default maximum 16 when enabled), including invalid records.
The explicit allocation budget defaults to 2048 MiB. It includes retained
compaction scratch and conservatively allows every transform to survive. Rescoring
adds selections, packed chemistry, ranges, scores, WeightedInteraction coverage
and the known vectors allocated by the ranker.
The shared receptor, CUDA context, host copies, wrapper-local query/scan scratch
and internal Thrust sort/reduction storage are excluded; this is not a total VRAM
limit. Batch size is a maximum: an explicit planning-budget failure splits a
contiguous group in half after temporary buffers unwind and retained compaction
scratch is released. A singleton gets one fresh-scratch retry, then fails with
its record ID and required/budget bytes. No pose or candidate is discarded.
CUDA allocation/device failures are fatal, rather than implicit retry triggers.
Aggregate arrays must also fit the backend's signed 32-bit count limit.

`gpu_docking` defaults to optimized host query preparation. The example fixes
`--query-method reference` explicitly to retain that protocol. Both methods
produce the same ordered queries; neither enables OpenMP without a thread request.

## Output and limitations

CSV/SDF export controls and per-record IDs retain their normal meaning.
`--pose-output all` exports the complete ranked post-prefilter pool.
Parsing and chemical-typing errors can be recorded with `--continue-on-error`.
CUDA/resource/output failures and undifferentiated ranker failures remain fatal;
completed output is retained. No record is retried through another backend.

`batches.csv` records aggregate counts, completed search/call times and a
conservative allocation estimate (`explicit_device_bytes`), not measured peak
VRAM. Search includes host packing, copies, device waits, download and cleanup.
The complete call includes rescoring and selection. Non-Fast batches append
rescoring, upload, scoring and ranking-stage times; these nested times are not
additive. Shared GPU search, score and upload times are blank in per-record
`timings.csv`. Preparation/query, prefilter, chemical typing and per-ligand
ranking/normalization/download durations remain separately attributable.
`summary.csv` records online wall time from input reading through final export,
excluding receptor startup and summary writing, and successful ligands/second.

`batch_attempts.csv` records actual subgroups, rejected attempts and completed
host intervals. Rejected work contributes to online and whole-call wall time,
but not successful-stage totals. With explicit batch overlap OFF/ON,
`batch_overlap_timeline.csv` pairs record query intervals with completed consumer
calls; summary fields include queue high-water estimates and producer/consumer
waits. These host spans contain host work as well as device waits. Use
`--overlap-trace` in a separate Nsight capture to check actual kernel overlap.

Non-Fast search downloads only compacted alignment IDs for segmentation and
keeps transforms, clash and Fast scores on device until rescoring. Ranked-row
downloads follow the requested export and reference-evaluation scope; full-pool
RMSD can retrieve all ranked poses even for Top-K export. Fast returns its full host pool.
Explicit overlap OFF/ON requires `--timings`, Quaternion and CSV-only output
(`--no-sdf` or `--pose-output none`). Paired RMSD references and per-record stage
profiling are excluded in that mode. A shared center-distance reference is
supported; see [output controls](output.md).

The [conformer experiment](../experiments/ligand_conformer_docking/README.md)
reuses batched search and raw scoring. Its default workflow normalizes Hybrid
over the generated ensemble on the CPU, with the crystallographic baseline kept
separate. Its `--diagnostics` mode compares original-included/excluded pools and
global Fast Top-500 subsets, each normalized independently. Optional conformer
batches default to four. This differs from per-ligand screening normalization.

## Tests and throughput comparisons

```sh
cmake --preset cuda-release -DENABLE_OPENMP=ON \
  -DDOCKING_TEST_COMPLEX=/path/to/1a1e
cmake --build --preset cuda-release
ctest --preset cuda-release \
  -R '^(cuda_ligand_batch(_rescoring)?_test|gpu_docking_batch.*_test)$'
```

Synthetic tests compare segmented clash/Fast, rescoring components and local
ranking against the original kernels. Real-data cases compare serial docking
with batches 1/2/4/8 for every method, complete selected pools, Top-10, partial
batches, errors and an empty index. Non-Fast cases check prefilter OFF/ON and
Hybrid normalization on intentionally different ligand populations.
`gpu_docking_batch_overlap_test` compares standard Vinardo without prefilter:
serial, batches 1/4/8/16 with OFF/ON prefetch, adaptive budgets and shutdown/error
paths. It requires a complex such as `1a1e` with substantial candidate pools.
These commands select the batch checks; see the [test guide](../tests/README.md)
for the complete suite. Compilation and test execution require a CUDA environment.
For throughput comparisons, use the same records, query threads, export policy
and timing scope, with new output directories and at least three repetitions.
Use the same binary and rescoring backend for batch OFF/ON comparisons; keep
CPU thread settings fixed when isolating overlap. Measure online wall and
end-to-end wall separately. Kernel sums and nested API timers do not reconstruct
either wall interval, and no throughput gain follows from batching alone.
