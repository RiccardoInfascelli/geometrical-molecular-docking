# Usage

Run these examples from the repository root, with the appropriate preset already built.
See the [build guide](build-e-test.md) for environment setup.

## Input files

Protein and pocket inputs are PDB files; ligands are MOL2 files, optionally containing multiple
records. Protein and pocket coordinates must share a frame. The starting ligand need not occupy
the pocket: docking determines its rotation and translation.

For CPU docking, a complex directory named `<id>` contains `<id>_protein.pdb`,
`<id>_pocket.pdb` and `<id>_ligand.mol2`. The screeners use the same protein/pocket naming
and accept a separate ligand file. `gpu_docking` accepts all three input paths explicitly.

Use a fresh output directory or CSV path for each run. Some applications overwrite existing files;
`gpu_docking` requires an output directory that does not yet exist.

## CPU docking

```sh
./build/cpu-release/geometric_docking /path/to/1a1e \
  --output /path/to/new-cpu-output
```

This single-complex pipeline scores every clash survivor with Fast, then applies WeightedInteraction
to the Fast top-500 and exports up to 20 ranked poses. Its protein index uses the ligand's
interaction-area diameter. It does not expose the CUDA application's ranking, cutoff or Top-K flags.

The output contains `benchmark.csv` and a `1a1e/` subdirectory with `ranking.csv`,
`pose_001.pdb`, subsequent pose files, copied reference structures and `view.pml` for PyMOL.
The textual report is written to stdout.

## CUDA docking

`gpu_docking` performs pose search on CUDA. This example ranks all clash survivors with Hybrid
using lambda 0.3 and Vinardo rep50:

```sh
./build/cuda-release/gpu_docking \
  /path/to/protein.pdb /path/to/pocket.pdb /path/to/ligands.mol2 \
  /path/to/new-gpu-output --limit 1 --ranking hybrid \
  --hybrid-lambda 0.3 --vinardo-alpha 0.5 --fast-prefilter none \
  --rescoring-backend cuda --top-k 10
```

`--ranking` accepts `fast`, `weighted-interaction`, `vinardo`, `vinardo-rep50` or `hybrid`.
Hybrid uses configurable `--hybrid-lambda` (default 0.3); it does not automatically imply rep50.
`--vinardo-alpha` defaults to 1. Setting it to 0.5 halves the repulsion contribution.
`--ranking vinardo-rep50` also fixes alpha at 0.5.

The main selection and rescoring options are:

| Option | Meaning |
| --- | --- |
| `--top-k K` | Final poses exported per record; default 20. |
| `--fast-prefilter none\|N` | Score all survivors, or only the Fast top-N, before rescoring; default `none`. |
| `--rescoring-backend cpu\|cuda` | Backend for WeightedInteraction, Vinardo-like and Hybrid; default `cpu`. Search remains CUDA. |
| `--interaction-region pocket\|protein` | Protein areas used by WeightedInteraction, including in Hybrid; default `pocket`. |

Vinardo-like always evaluates the full protein's supported heavy atoms, independently of
`--interaction-region`. Top-K affects final export; a Fast prefilter changes the rescoring pool.

Defaults are Fast ranking, no record limit, a 24 Å shared triplet cutoff and SVD alignment.
Use `--cutoff A` to change the cutoff or `--kabsch quaternion` to select quaternion alignment.
The example above overrides the ranking and rescoring defaults.

## Multi-ligand screening

Both screeners report the best Fast score per ligand, rather than a full rescored ranking.
Use the same explicit record limit and cutoff when comparing them:

```sh
./build/cpu-release/multi_ligand_screen \
  /path/to/1a1e /path/to/ligands.mol2 /path/to/new-cpu-screen.csv \
  --limit 100 --cutoff 24

./build/cuda-release/multi_ligand_screen_gpu \
  /path/to/1a1e /path/to/ligands.mol2 /path/to/new-gpu-screen.csv \
  --limit 100 --cutoff 24
```

`--limit N` selects the first N records, not a random sample. The defaults are 500 records on CPU
and 30 on CUDA; both use a 24 Å cutoff. Protein preparation and a single shared index are reused
across ligands, including those whose diameter exceeds the cutoff. No per-ligand fallback occurs.
The cutoff limits indexed triplet sides and can reduce candidate coverage.

The CUDA screener optionally exports joint scores with `--rescore-output NEW_DIR`.
Search and Fast remain on CUDA; WeightedInteraction and Vinardo-like are calculated on CPU.
This export retains survivor order and does not provide Hybrid ranking.

## Outputs

| Application | Main outputs |
| --- | --- |
| `geometric_docking` | `benchmark.csv`, per-complex ranking CSV, ranked PDB poses and PyMOL script. |
| `gpu_docking` | `poses.csv`, `record-N.sdf`, `records.csv`, `summary.csv`; `timings.csv` with `--timings`. |
| Both screeners | The requested CSV: per-record status, candidate/pose counts, best Fast score and timings. |

In CUDA docking, `poses.csv` stores selected poses, transforms, clash information and calculated
scores. `rank` is the final position; `pose_id`, when present, identifies the original survivor.
`records.csv` reports each ligand's status and counts; `summary.csv` records the completed run's
configuration. In rescoring exports, unused score fields remain blank.

Fast, WeightedInteraction and Hybrid are maximized; Vinardo-like is minimized.
For repulsion reweighting, `vinardo_like_score` is the original total and
`vinardo_adjusted_score` is the value used for ranking. Hybrid normalizes over each record's
rescoring pool before Top-K selection; values from different pools are not directly comparable.
Vinardo-like values are not validated binding affinities in kcal/mol.

For CUDA RMSD evaluation, add `--reference-mol2 /path/to/reference.mol2`. References are paired in
record order and must match each ligand's atom order and molecular graph. Reference poses must use
the protein frame. The export reports heavy-atom RMSD after ranking, without superposition or
symmetry correction. CPU diagnostics instead compare all retained atoms against the input conformer:
this measures crystal-reference error only when the input is that crystal pose. RMSD is never used
for ranking.

Check statuses and errors before interpreting results. Screening `ok` can still mean zero
survivors; a missing score or RMSD is not zero. Preserve stdout, stderr and exit status.
CPU and CUDA timing fields do not always cover identical work and are not always directly comparable.
For score definitions and scientific limitations, see the
[algorithm guide](come-funziona-il-docking.md).
