#!/usr/bin/env python3
"""Compare disabled target pruning with one clearance threshold on crystal complexes.

The input list contains one complex ID per line (blank lines and # comments are
ignored). Each directory must contain its own <id>_protein.pdb, <id>_pocket.pdb
and crystallographic <id>_ligand.mol2. No dataset discovery or copying is done.
Only the Python standard library is required.

RMSD follows the executable: all input atoms, atom-index correspondence, no refit
or symmetry correction. This is best-available RMSD, not a ranked Top-1 metric.
Process, format and numerical failures give a nonzero exit status after all
complexes have been attempted; no poses is recorded as a separate outcome.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time


FIELDS = (
    "clearance", "targets", "protein_triplets", "candidate_matches", "transforms",
    "survivors_5pct", "best_rmsd_5pct", "best_rmsd_10pct", "index_s", "search_s",
    "targets_removed", "protein_prep_s", "clearance_s", "ligand_prep_s", "filter_s",
)
COUNTS = (
    "targets", "protein_triplets", "candidate_matches", "transforms",
    "survivors_5pct", "targets_removed",
)
COMMON_TIMES = ("protein_prep_s", "clearance_s", "ligand_prep_s")
TIMES = COMMON_TIMES + ("filter_s", "index_s", "search_s")
REDUCTIONS = COUNTS[:-1] + ("index_s", "search_s", "index_search_s")
ROW_FIELDS = COUNTS + (
    "best_rmsd_5pct", "best_rmsd_10pct", "filter_s", "index_s", "search_s",
)
DERIVED_FIELDS = (
    "baseline_index_search_s", "pruned_index_search_s",
) + tuple(f"{field}_reduction_pct" for field in REDUCTIONS) + (
    "index_search_speedup", "best_rmsd_5pct_delta_A",
)
CSV_FIELDS = (
    "complex_id", "complex_path", "status", "error", "exit_code", "threshold_A",
    "baseline_status", "pruned_status", "process_wall_s", "stdout_log", "stderr_log",
) + COMMON_TIMES + tuple(
    f"{side}_{field}" for side in ("baseline", "pruned") for field in ROW_FIELDS
) + DERIVED_FIELDS


def parse_output(stdout, threshold=1.25):
    """Read the two TSV rows; retain raw RMSD tokens such as inf and nan.

    A failed process may have produced just one row. Return it for inspection;
    the caller decides whether the complete comparison succeeded.
    """
    rows = {}
    found_header = False
    for line in stdout.splitlines():
        values = line.split("\t")
        if values[0] == "clearance":
            if found_header or tuple(values) != FIELDS:
                raise ValueError("Unexpected or repeated comparison header")
            found_header = True
            continue
        if not found_header or "\t" not in line:
            continue
        if len(values) != len(FIELDS):
            raise ValueError("Comparison row has an unexpected number of columns")
        row = dict(zip(FIELDS, values))
        if row["clearance"] == "disabled":
            side = "disabled"
        else:
            value = float(row["clearance"])
            # The executable prints thresholds with six decimal places.
            if not math.isclose(value, threshold, rel_tol=0, abs_tol=0.00000051):
                raise ValueError("Output threshold differs from the requested threshold")
            side = "pruned"
        if side in rows:
            raise ValueError(f"Repeated {side} comparison row")
        for field in COUNTS:
            if int(row[field]) < 0:
                raise ValueError(f"Negative count in {field}")
        for field in TIMES:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid time in {field}")
        for field in ("best_rmsd_5pct", "best_rmsd_10pct"):
            float(row[field])
        rows[side] = row
    if not found_header:
        raise ValueError("Comparison header not found; rebuild the experiment executable")
    return rows


def pose_status(row):
    if row is None:
        return "missing"
    if int(row["survivors_5pct"]) == 0:
        return "no_poses"
    # A 5%-clash survivor is also within the relaxed 10% diagnostic limit.
    for field in ("best_rmsd_5pct", "best_rmsd_10pct"):
        rmsd = float(row[field])
        if not math.isfinite(rmsd) or rmsd < 0:
            return "invalid_rmsd"
    return "ok"


def compare_rows(baseline, pruned):
    if int(baseline["targets_removed"]) != 0:
        raise ValueError("Disabled pruning unexpectedly removed targets")
    if int(baseline["targets"]) != int(pruned["targets"]) + int(pruned["targets_removed"]):
        raise ValueError("Retained and removed targets do not match the baseline")
    for field in COMMON_TIMES:
        if float(baseline[field]) != float(pruned[field]):
            raise ValueError(f"Shared preparation time differs between rows: {field}")
    values = {}
    for side, row in (("baseline", baseline), ("pruned", pruned)):
        values[f"{side}_index_search_s"] = float(row["index_s"]) + float(row["search_s"])
    for field in REDUCTIONS:
        before = values["baseline_index_search_s"] if field == "index_search_s" else float(baseline[field])
        after = values["pruned_index_search_s"] if field == "index_search_s" else float(pruned[field])
        values[f"{field}_reduction_pct"] = 100 * (before - after) / before if before > 0 else None
    before, after = values["baseline_index_search_s"], values["pruned_index_search_s"]
    values["index_search_speedup"] = before / after if after > 0 else None
    values["best_rmsd_5pct_delta_A"] = (
        float(pruned["best_rmsd_5pct"]) - float(baseline["best_rmsd_5pct"])
        if pose_status(baseline) == pose_status(pruned) == "ok" else None
    )
    return values


def run_one(executable, complex_dir, threshold, output, timeout):
    complex_id = complex_dir.name
    stdout_path = output / f"{complex_id}.stdout.log"
    stderr_path = output / f"{complex_id}.stderr.log"
    command = [str(executable), str(complex_dir), "--compare-clearance", str(threshold)]
    row = {
        "complex_id": complex_id, "complex_path": str(complex_dir),
        "threshold_A": threshold, "stdout_log": stdout_path.name,
        "stderr_log": stderr_path.name, "status": "ok", "error": "", "exit_code": "",
    }
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        start = time.perf_counter()
        try:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            row["exit_code"] = result.returncode
            if result.returncode != 0:
                row.update(status="failed", error=f"Experiment exited with code {result.returncode}; see stderr log")
        except subprocess.TimeoutExpired:
            row.update(status="timeout", error=f"Experiment exceeded {timeout} seconds")
        except OSError as error:
            row.update(status="failed", error=str(error))
        # Includes common preparation and BOTH searches, not one variant.
        row["process_wall_s"] = time.perf_counter() - start
    parsed = {}
    try:
        parsed = parse_output(stdout_path.read_text(encoding="utf-8", errors="replace"), threshold)
        if row["status"] == "ok":
            if set(parsed) != {"disabled", "pruned"}:
                raise ValueError("Expected both disabled and pruned comparison rows")
            row.update(compare_rows(parsed["disabled"], parsed["pruned"]))
    except (ValueError, OverflowError) as error:
        if row["status"] == "ok":
            row.update(status="invalid_output", error=str(error))
        else:
            row["error"] += f"; output: {error}"
    for side, key in (("baseline", "disabled"), ("pruned", "pruned")):
        data = parsed.get(key)
        row[f"{side}_status"] = pose_status(data)
        if data is not None:
            row.update({f"{side}_{field}": data[field] for field in ROW_FIELDS})
    available = parsed.get("disabled") or parsed.get("pruned")
    if available:
        row.update({field: available[field] for field in COMMON_TIMES})
    if row["status"] == "ok":
        states = {row["baseline_status"], row["pruned_status"]}
        if "invalid_rmsd" in states:
            row["status"] = "invalid_rmsd"
        elif "no_poses" in states:
            row["status"] = "no_poses"
    return row


def read_complex_ids(path):
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        if value in (".", "..") or Path(value).name != value:
            raise ValueError(f"Expected a complex ID, not a path: {value}")
        if value in ids:
            raise ValueError(f"Repeated complex ID: {value}")
        ids.append(value)
    if not ids:
        raise ValueError("The complex list is empty")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--complex-root", type=Path, required=True)
    parser.add_argument("--complex-list", type=Path, required=True, help="One crystallographic complex ID per line")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--timeout", type=float, help="Optional timeout in seconds per complex")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and non-negative")
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be finite and positive")
    executable = args.executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error(f"Executable not found or not executable: {executable}")
    try:
        complex_ids = read_complex_ids(args.complex_list)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"Cannot create a fresh output directory: {error}")
    root = args.complex_root.resolve()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(executable), "complex_root": str(root),
        "complex_list": str(args.complex_list.resolve()), "complex_ids": complex_ids,
        "threshold_A": args.threshold, "timeout_s": args.timeout,
        "command_template": shlex.join([str(executable), "<complex_directory>", "--compare-clearance", str(args.threshold)]),
        "environment": {key: value for key, value in sorted(os.environ.items()) if key.startswith("DOCKING_")},
        "rmsd": "All input atoms, atom-index correspondence, crystal input conformer; no refit or symmetry correction",
        "quality_metric": "Best available RMSD among survivors at clash <= 5%; no scoring or ranked Top-k",
        "index_cutoff": "Ligand diameter, not the shared screening cutoff of 24 A",
        "order": "Serial complexes in list order; disabled search first, pruned search second in each process",
        "times": "Common preparation reported once; search_s includes grid support and RMSD diagnostics; process_wall_s includes both variants",
        "reductions": "100 * (baseline - pruned) / baseline; blank when baseline is zero",
        "rmsd_delta": "Pruned minus baseline, positive is worse; blank if either variant has no poses or invalid RMSD",
        "invalid_values": "Raw inf/nan retained; unavailable derived values are blank, never converted to zero",
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    statuses = Counter()
    with (output / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        stream.flush()
        for number, complex_id in enumerate(complex_ids, 1):
            row = run_one(executable, root / complex_id, args.threshold, output, args.timeout)
            writer.writerow({key: format(value, ".9g") if isinstance(value, float) else value
                             for key, value in row.items()})
            stream.flush()
            statuses[row["status"]] += 1
            print(f"[{number}/{len(complex_ids)}] {complex_id}: {row['status']}", flush=True)
    print(f"Results: {output / 'comparison.csv'}")
    print("Statuses: " + ", ".join(f"{key}={count}" for key, count in sorted(statuses.items())))
    return int(any(statuses[state] for state in ("failed", "timeout", "invalid_output", "invalid_rmsd")))


if __name__ == "__main__":
    raise SystemExit(main())
