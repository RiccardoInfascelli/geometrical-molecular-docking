# CPU conformer-quality experiment

`conformer_quality` measures how closely ETKDGv3 samples the crystallographic
ligand conformation. It does not run docking, use the protein, or calculate any
interaction score. Results measure conformational coverage after free rigid
alignment, not placement accuracy in a binding pocket.

## Protocol

For each ligand, one RDKit C++ `EmbedMultipleConfs` call requests at most 200
conformers, with a fixed nonnegative seed and one thread. There is no MMFF,
energy sorting, RMSD pruning, diversity filtering, or replacement of failed
embeddings. Returned conformers retain their order. Budgets 1, 5, 10, 20, 50,
100 and 200 use prefixes of this same saved sequence; they never regenerate it.

Evaluation fits all corresponding heavy atoms using Horn's quaternion matrix
and its largest eigenvector. This is a symmetric 4x4 eigensolve, **not SVD**,
and permits only proper rotations. Atom correspondence is fixed by the original
MOL2 atom order and checked against both parsed graphs. There is no symmetry
permutation, so equivalent atom labels may increase the measured RMSD. Distances
are in angstroms. Success uses strict `< 1`, `< 1.5` and `< 2` thresholds.

### Keeping crystal coordinates out of generation

The generation input is parsed from a separate in-memory MOL2 block whose XYZ
fields have already been replaced by zero. All conformers and stereochemical
flags are then removed. The independently parsed original supplies only the
evaluation coordinates; its stereo flags are never transferred to generation.
Atom identities, names, original file IDs, graph connectivity, charges and
ordering are checked before evaluation. Working hydrogens added by RDKit are
not added to the saved original-atom mapping.

MOL2 often specifies chirality only through 3D geometry. Reusing that information
would leak the reference into this experiment. By default, a graph with potential
undefined stereochemistry is recorded as `stereo_ambiguous` without generating
conformers. Detection uses the installed RDKit version's `findPotentialStereo`.
This limits dataset coverage: the current protocol does not accept an external
independently mapped stereochemical specification.

`--allow-undefined-stereo` explicitly permits graph-only generation of these
cases. It can explore different stereoisomers, so those results must remain
separate from a study of a fixed stereoisomer. It never restores native stereo.
The count of undefined stereo elements and parser-defined native stereo flags
is recorded. The latter is an audit count, not a generation constraint.

## Build and a controlled batch

Inside the configured CPU environment, from the repository root:

```sh
cmake -S experiments/ligand_conformer_docking -B build/conformer-host \
  -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DTARGET_CPU_ARCHITECTURE=generic
cmake --build build/conformer-host --target conformer_quality --parallel 1
```

For the synthetic geometry and runner checks:

```sh
cmake --build build/conformer-host --target conformer_quality conformer_quality_smoke --parallel 1
ctest --test-dir build/conformer-host -R '^conformer_quality_(smoke|runner)$' --output-on-failure
```

The runner test includes mocked cases and a real two-conformer synthetic smoke,
enabled through `CONFORMER_QUALITY_EXECUTABLE` by CTest. It checks CSV files,
ambiguous stereochemistry and resume without duplication. Its total CTest
timeout is 30 seconds; it does not read PDBbind or launch a dataset campaign.

Prepare a text file of explicitly chosen complex IDs, one per line. Each ID
resolves to `<dataset-root>/<id>/<id>_ligand.mol2`; absolute complex directories
are also accepted. Blank lines and lines beginning with `#` are ignored.
The runner does not enumerate the dataset or launch any work without this list.

```sh
python3 -B experiments/ligand_conformer_docking/run_conformer_quality.py \
  --executable build/conformer-host/conformer_quality \
  --dataset-root /path/to/refined-set --complex-list /path/to/approved-ids.txt \
  --max-conformers 200 --seed 20261027 --limit 3 \
  --output experiments/ligand_conformer_docking/runs/quality-first-001
```

The output directory must not exist. The same explicit seed is used for each
ligand; reproducibility also requires the recorded RDKit/compiler environment
and the same executable. The maximum request must stay fixed when comparing
budgets: a separate run requesting fewer conformers is a different protocol.

To resume, use the identical command with `--resume` in place of `--output`.
The runner checks input order, seed, options, executable SHA-256 and targeted
input file size/mtime. Completed results are reused, including threshold failures,
incomplete generation and terminal error summaries. An interrupted attempt with
incomplete files is preserved and a new numbered attempt is created. Changing
the protocol requires a new output directory. Do not run two writers on one batch.

## Outputs and interpretation

| File | Contents |
| --- | --- |
| Batch `summary.csv` | One row per case with an intact result: status, best RMSD, first successful sequence indices, counts and times. |
| Batch `budgets.csv` | Seven rows per case: available prefix size, completeness, best observed RMSD and threshold indicators. |
| `failures.csv` | Incomplete, rejected or erroneous cases, with their separate status and process exit information. |
| `manifest.json` | Ordered input list, executable identity, seed and run options; resume compatibility record. |
| Per-case `conformers.csv` | One-based sequence index, original RDKit conformer ID, aligned RMSD and threshold indicators. |
| Per-case `conformer_coordinates.csv` | Generated coordinates of every original atom, linked to sequence index. |
| Per-case `atom_mapping.csv` | Zero-based original index, MOL2 atom ID/name, atomic number and heavy-atom evaluation index. |
| Per-case `protocol.json` | RDKit/compiler versions, ETKDG settings, stereo policy, alignment and evaluation conventions. |

The optional `--no-coordinates` flag, available in both the executable and the
runner, omits `conformer_coordinates.csv` entirely. It skips file creation and
coordinate formatting, while preserving generation, every aligned RMSD,
`conformers.csv`, `atom_mapping.csv`, all budget metrics and the summary.
Coordinates remain enabled by default. The engine protocol records the flag
and coordinate-file size; the runner includes it in resume compatibility.
Changing it requires a new batch directory. An older manifest without this
field retains its historical default of exporting coordinates and is never
rewritten during resume.

This experiment has no pose-ranking Top-K: budgets are prefixes of the same
ETKDG sequence. Suppressing coordinates does not reduce any prefix or choose
conformers by RMSD. Without the coordinate file, independent geometric replay
of the generated conformers is unavailable. `evaluation_s` still includes the
remaining conformer-metric CSV writes, so it is not a pure eigensolve timer.

Per-case files and stdout/stderr remain under `cases/<case>/attempt-NNN/result/`
(logs are beside `result/`). Aggregate tables are atomically refreshed from
these retained results; no input or previous attempt is overwritten.

`status=ok` means the requested sequence was generated and evaluated, not that
RMSD was successful. A zero threshold indicator is an observed failure to reach
that threshold. `incomplete` means fewer conformers were available than requested;
`generation_failed`, `stereo_ambiguous` and `error` remain distinct. Empty RMSD or
first-hit cells mean unavailable/not reached, never zero RMSD. First-hit indices
are one-based and refer to returned valid conformers, not failed embedding trials.

Each budget reports its available prefix even if incomplete. Interpret success
only alongside `complete` and `status`; a smaller available sample is not a full
budget. Budgets above an explicitly smaller maximum have `not_requested` status.
Report failure/incomplete/rejection denominators separately rather than dropping
those cases from the dataset. Preparation, generation and evaluation/export
timers are recorded separately; app wall time ends before final metadata flush,
and the runner also records process wall time in each attempt's `status.json`.

The tiny smoke checks use synthetic geometry and two-conformer sequences. They
do not establish scientific validity on PDBbind or performance at large budgets.
