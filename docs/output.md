# Output controls

Export controls select what is saved after pose search and scoring. They do not
change Kabsch, clash, scoring formulas, Hybrid normalization, tie-breaking or the
Fast prefilter. Use a new output directory for every run; reducing export does
not remove files from an earlier run.

## Pose quantity and detail

The operational CPU and CUDA docking applications support these export options:

| Option | Meaning |
| --- | --- |
| `--pose-output none` | Do not create pose CSV or structural pose files. Scientific summaries remain. |
| `--pose-output topk --top-k K` | Save the first K poses of each available ranking. |
| `--pose-output all` | Save the whole available ranking. `--top-k` does not truncate this export. |
| `--pose-detail compact` | Save IDs/ranks, scores, RMSD when available and essential clash metrics. |
| `--pose-detail full` | Include the available detailed fields and rigid transforms R/t. |

CPU and CUDA operational docking default to Top-20 export. Without an explicit
detail option, each application uses its default CSV schema. CPU `full` adds
transforms to `ranking.csv`. Top-K applies per record and does not deduplicate poses.

`all` means the pool actually evaluated by that executable. CPU docking uses its
existing Fast top-500 before WeightedInteraction. GPU rescoring may use an explicit
`--fast-prefilter N`. Export cannot recover poses excluded by such a prefilter.
To rescore every GPU clash survivor, use `--fast-prefilter none`; this is a
scientific protocol choice, independent of the output quantity.

## Operational CPU docking

```sh
./build/cpu-release/geometric_docking /path/to/complex \
  --output /path/to/new-output --pose-output none --quiet --timings

./build/cpu-release/geometric_docking /path/to/complex \
  --output /path/to/new-ranking --pose-output all --pose-detail full --no-structures
```

`benchmark.csv` retains the existing counts, ranking diagnostics and RMSD metrics.
`--no-structures` suppresses reference copies, PDB poses and PyMOL files while
retaining `ranking.csv`. `none` skips pose serialization. An enabled center-distance
diagnostic still loads and validates its reference during output handling.

## Operational CUDA docking

```sh
./build/cuda-release/gpu_docking protein.pdb pocket.pdb ligands.mol2 \
  /path/to/new-benchmark --limit 100 --kabsch quaternion \
  --ranking fast --pose-output none --timings --diagnostics off

./build/cuda-release/gpu_docking protein.pdb pocket.pdb ligands.mol2 \
  /path/to/new-full-pool --ranking hybrid --hybrid-lambda 0.3 --vinardo-alpha 0.5 \
  --fast-prefilter none --pose-output all --pose-detail full --no-sdf
```

`--no-sdf` selects CSV-only pose export. `none`
suppresses SDF automatically, without disabling computation or ranking.
`records.csv` and `summary.csv` remain; `timings.csv` requires `--timings`.
`exported_poses` is the number saved, not the survivor count or scoring-pool size.

With `--pose-output` or `--pose-detail`, `pose_metrics.csv` contains one compact summary per
successful record, including the Top-1 identity and calculated scores. Fast uses
`alignment_index` as its available stable identity; its separate `pose_id` is blank.
Without `--reference-mol2`, RMSD fields are empty, not zero.

With a reference and either explicit export option, every final-ranked pose is retrieved
and evaluated before export selection. The metric summary retains minimum RMSD
in Top-1/5/10/20 and the best RMSD of that **post-prefilter ranked pool** in all
output modes. It is not an oracle before the prefilter. This evaluation costs
additional downloads and RMSD work, even with `none`. Without explicit export
controls, evaluation covers only the returned/exported poses.
RMSD remains direct heavy-atom RMSD with fixed mapping, never a ranking criterion.

The `all` mode can increase sorting, download and buffering costs as well as I/O.
Keep the evaluation scope fixed when comparing timings. Use `--no-sdf` when
molecular structures are unnecessary.

## Center-distance diagnostic

Both docking applications accept
`--pose-center-distance on --center-reference-mol2 /path/to/reference.mol2`.
This adds `center_distance_A` to CPU `ranking.csv`, or CUDA `poses.csv` and SDF
properties. It is the Euclidean distance between the transformed ligand's
heavy-atom centroid and the reference centroid in the protein frame. It measures position, without atom
mapping, alignment or an RMSD calculation; it never changes ranking.
The shared reference must contain exactly one molecule. CPU docking requires
this file explicitly; CUDA reads it once before device setup and can instead
use `--reference-mol2` for paired references. Paired-reference compatibility
restrictions still apply. Shared references support CUDA batching, overlap and
`--continue-on-error` without requiring matching ligand graphs.

The default is `--pose-center-distance off`: schemas are unchanged and the shared
center-reference file is not read. Only the exported prefix is evaluated when ON;
`none` calculates no pose distances and does not request a larger ranked pool.

## Offline analysis

Joint scoring exports, scoring ablation and the older offline Hybrid conformer
workflow still require their complete inputs. Their output schemas are unchanged.
Do not substitute `compact`, `topk` or `none` files for a declared complete pool.
Replay requires all pose identities/transforms plus the exact original conformer
coordinates and atom mapping, receptor/reference inputs and scientific protocol.
The rescoring comparison script explicitly rejects missing
poses or compact exports and supports full `all` exports without treating Top-K
as a prefilter.

## Timing and output size

CPU `--timings` writes `compute_s`, `export_s` and `wall_before_report_s`; the last
ends before timing-file serialization, the report and process teardown. GPU
timings retain their application-specific scopes. With explicit export controls,
`reference_evaluation_s` and `pose_export_s` split the existing combined
`reference_rmsd_export_s`; do not sum the split intervals with their parent.
The pose-export interval includes the small metric-summary write. Record and
run-summary I/O also contribute to wall time.

For any application, measure the final fresh directory size
separately, for example with `du -sk /path/to/new-output`, and capture external
process wall time when startup, final flushes and destruction matter.
The two screeners emit compact per-record summaries and do not accept pose-export
or center-distance controls. Diagnostic stdout and optional stage profiling are
separate from these export settings.
