#!/usr/bin/env bash
set -euo pipefail

# Generate one complete pool with the existing driver, then replay scoring only.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
complex="${1:-/workspace/datasets/refined-set/1a1e}"
run="${2:-$script_dir/runs/1a1e-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
build="${3:-$script_dir/build}"
complex="$(cd -- "$complex" && pwd)"
complex_id="$(basename -- "$complex")"
ligand="$complex/${complex_id}_ligand.mol2"
generator="$build/engine/vinardo_rescore_experiment"
ablation="$build/scoring_ablation"

for input in "$complex/${complex_id}_protein.pdb" "$complex/${complex_id}_pocket.pdb" "$ligand"; do
    if [[ ! -r "$input" ]]; then
        printf 'Input is not readable: %s\n' "$input" >&2
        exit 1
    fi
done
for executable in "$generator" "$ablation"; do
    if [[ ! -x "$executable" ]]; then
        printf 'Executable not found: %s\nBuild the isolated experiment first.\n' "$executable" >&2
        exit 1
    fi
done
if [[ -e "$run" ]]; then
    printf 'Output already exists: %s\n' "$run" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$run")"
mkdir -- "$run"
run="$(cd -- "$run" && pwd)"

generation=(env -u DOCKING_PROFILE_STAGES -u DOCKING_STAGE_TIMINGS -u DOCKING_KABSCH_TIMINGS
    DOCKING_KABSCH_METHOD=quaternion DOCKING_CPU_CLASH_METHOD=cached DOCKING_CPU_SCREENING_ONLY=1
    "$generator" "$complex" --joint-output "$run/generation" --cutoff 24 --interaction-region pocket)
printf '%q ' "${generation[@]}" > "$run/commands.txt"
printf '\n' >> "$run/commands.txt"
"${generation[@]}" > "$run/generation.stdout" 2> "$run/generation.stderr"

pool_size="$(python3 - "$run/generation" <<'PY'
import csv
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
with (directory / 'protocol.csv').open(newline='') as source:
    protocol = dict(csv.reader(source))
if protocol.get('pose_pool') != 'all_clash_survivors' or protocol.get('fast_top_n') != 'none':
    raise SystemExit('Generation applied a prefilter; refusing to label this as the complete pool')
expected = int(protocol['input_survivors'])
if int(protocol['exported_poses']) != expected:
    raise SystemExit('Generation did not export all clash survivors')
with (directory / 'poses.csv').open(newline='') as source:
    rows = list(csv.DictReader(source))
if len(rows) != expected or any(row['record_index'] != '1' for row in rows):
    raise SystemExit('CSV row count or record differs from the generation protocol')
if len({row['pose_id'] for row in rows}) != expected:
    raise SystemExit('Duplicate pose IDs in generated pool')
print(expected)
PY
)"

analysis=("$ablation" --complex "$complex" --poses "$run/generation/poses.csv"
    --pool-size "$pool_size" --pool-kind all-survivors --interaction-region pocket
    --reference "$ligand" --vinardo-alpha 1 --hybrid-lambda 0.3 --output "$run/ablation")
printf '%q ' "${analysis[@]}" >> "$run/commands.txt"
printf '\n' >> "$run/commands.txt"
"${analysis[@]}" > "$run/ablation.stdout" 2> "$run/ablation.stderr"
printf 'Complete pool: %s poses\nOutput: %s\n' "$pool_size" "$run"
