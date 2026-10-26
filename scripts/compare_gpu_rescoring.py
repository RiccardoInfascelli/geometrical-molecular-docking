#!/usr/bin/env python3
"""Compare completed gpu_docking CPU/CUDA rescoring exports, without running docking.

Use identical settings and a top-k large enough to export the full selected pool
when validating all scores. RMSD never participates in comparison or ordering.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys


TRANSFORM = "r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz".split()
SCORES = "weighted_interaction_score vinardo_like_score vinardo_gaussian vinardo_repulsion vinardo_hydrophobic vinardo_hydrogen_bond vinardo_adjusted_score hybrid_score".split()
POSE_FIELDS = ["record_index", "molecule_id", "rank", "alignment_index",
               "fast_weighted_score", "clash_fraction", "occupied_atoms", "pose_id"] + TRANSFORM + SCORES
RECORD_FIELDS = "record_index molecule_id status diameter_A diameter_gt_cutoff ligand_triplets candidate_matches alignments valid_transforms survivors exported_poses sdf candidate_pool".split()
BACKEND_FIELDS = {"rescoring_backend", "survivor_download_policy", "hybrid_precision"}


def read_csv(path, required):
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or not set(required) <= set(fields):
                raise ValueError(f"Invalid or missing columns: {path}")
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Incomplete CSV row: {path}")
    return fields, rows


def integer(value):
    number = int(value)
    if number < 0:
        raise ValueError(f"Negative ID or count: {value}")
    return number


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite numeric value: {value}")
    return number


def load_run(directory, backend):
    _, summary_rows = read_csv(directory / "summary.csv", ["key", "value"])
    summary = {}
    for row in summary_rows:
        if row["key"] in summary:
            raise ValueError(f"Duplicate summary key: {row['key']}")
        summary[row["key"]] = row["value"]
    required = {"status", "processed", "top_k", "ranking_method", "rescoring_backend",
                "pose_pool", "fast_prefilter", "index_policy", "index_cutoff_A", "tie_break"}
    if not required <= summary.keys() or summary["status"] != "complete":
        raise ValueError(f"Missing complete summary/protocol: {directory}")
    if summary["rescoring_backend"] != backend:
        raise ValueError(f"Expected {backend} rescoring export: {directory}")
    if integer(summary["top_k"]) == 0:
        raise ValueError("top_k must be positive")
    for key, value in summary.items():
        if (key.endswith("_A") or key.endswith("_s") or "weight_" in key or
                key in {"vinardo_alpha", "hybrid_lambda", "operational_clash_fraction"}):
            if value not in {"off", "not_calculated"}:
                finite(value)
    _, record_rows = read_csv(directory / "records.csv", RECORD_FIELDS)
    records = {}
    for row in record_rows:
        index = integer(row["record_index"])
        if index in records:
            raise ValueError(f"Duplicate record ID: {index}")
        if row["status"] not in {"ok", "no_poses", "no_triplets"}:
            raise ValueError(f"Unsuccessful record {index}: {row['status']}")
        for field in RECORD_FIELDS:
            if field not in {"record_index", "molecule_id", "status", "diameter_A", "sdf"}:
                integer(row[field])
        finite(row["diameter_A"])
        records[index] = row
    if not records or len(records) != integer(summary["processed"]):
        raise ValueError(f"Incomplete record export: {directory}")
    fields, pose_rows = read_csv(directory / "poses.csv", POSE_FIELDS)
    poses, orders = {}, {index: [] for index in records}
    for row in pose_rows:
        index, pose_id = integer(row["record_index"]), integer(row["pose_id"])
        key = (index, pose_id)
        if index not in records or key in poses:
            raise ValueError(f"Unknown record or duplicate pose ID: {key}")
        if integer(row["rank"]) != len(orders[index]) + 1:
            raise ValueError(f"Ranks are not contiguous in export order: {key}")
        if row["molecule_id"] != records[index]["molecule_id"]:
            raise ValueError(f"Molecule identity mismatch: {key}")
        for field in ["alignment_index", "occupied_atoms"]:
            integer(row[field])
        if pose_id >= integer(records[index]["survivors"]):
            raise ValueError(f"Pose ID outside original survivor pool: {key}")
        for field in ["fast_weighted_score", "clash_fraction"] + TRANSFORM + SCORES:
            if row[field] != "" or field not in SCORES:
                finite(row[field])
        if "rmsd_heavy_A" in row:
            finite(row["rmsd_heavy_A"])
        method = summary["ranking_method"]
        needed = (["weighted_interaction_score"] if method in {"weighted-interaction", "hybrid"} else [])
        if method in {"vinardo", "hybrid"}:
            needed += SCORES[1:-1]
        if method == "hybrid":
            needed += ["hybrid_score"]
        if method not in {"weighted-interaction", "vinardo", "hybrid"} or any(not row[f] for f in needed):
            raise ValueError(f"Missing scores for requested rescoring method: {key}")
        poses[key] = row
        orders[index].append(pose_id)
    for index, row in records.items():
        count = len(orders[index])
        if (count != integer(row["exported_poses"]) or
                count != min(integer(summary["top_k"]), integer(row["candidate_pool"])) or
                integer(row["candidate_pool"]) > integer(row["survivors"])):
            raise ValueError(f"Incomplete pose export for record {index}")
        if summary["fast_prefilter"] == "none" and row["candidate_pool"] != row["survivors"]:
            raise ValueError(f"Unfiltered scoring must include every survivor for record {index}")
        if (row["status"] == "ok") != (count > 0):
            raise ValueError(f"Status and pose count disagree for record {index}")
    return summary, records, poses, orders, set(fields) - {"rmsd_heavy_A"}


def compare_runs(cpu_dir, cuda_dir):
    cpu, gpu = load_run(Path(cpu_dir), "cpu"), load_run(Path(cuda_dir), "cuda")
    report = {key: [] for key in ("protocol_mismatches", "record_mismatches", "pose_set_mismatches",
              "numeric_mismatches", "top1_mismatches", "topk_set_mismatches", "order_mismatches")}
    report["max_errors"] = {}
    for key in sorted(cpu[0].keys() | gpu[0].keys()):
        if key not in BACKEND_FIELDS and not key.endswith("_s") and cpu[0].get(key) != gpu[0].get(key):
            report["protocol_mismatches"].append(key)
    if cpu[4] != gpu[4]:
        report["protocol_mismatches"].append("poses.csv schema")
    for index in sorted(cpu[1].keys() | gpu[1].keys()):
        if cpu[1].get(index) != gpu[1].get(index):
            report["record_mismatches"].append(index)
        before, after = cpu[3].get(index, []), gpu[3].get(index, [])
        if before[:1] != after[:1]:
            report["top1_mismatches"].append(index)
        if set(before) != set(after):
            report["topk_set_mismatches"].append(index)
        if before != after:
            report["order_mismatches"].append(index)
    for key in sorted(cpu[2].keys() ^ gpu[2].keys()):
        report["pose_set_mismatches"].append(list(key))
    numeric = ["fast_weighted_score", "clash_fraction"] + TRANSFORM + SCORES
    for key in sorted(cpu[2].keys() & gpu[2].keys()):
        old, new = cpu[2][key], gpu[2][key]
        for field in ["molecule_id", "alignment_index", "occupied_atoms"]:
            if old[field] != new[field]:
                report["numeric_mismatches"].append([*key, field])
        for field in numeric:
            if old[field] == new[field] == "":
                continue
            if not old[field] or not new[field]:
                report["numeric_mismatches"].append([*key, field])
                continue
            a, b = finite(old[field]), finite(new[field])
            delta, scale = abs(a - b), max(abs(a), abs(b))
            errors = report["max_errors"].setdefault(field, {"absolute": 0.0, "relative": 0.0})
            errors["absolute"] = max(errors["absolute"], delta)
            errors["relative"] = max(errors["relative"], delta / scale if scale else 0.0)
            rel, absolute = (1e-9, 1e-11) if field == "hybrid_score" else (
                (1e-8, 1e-10) if field.startswith("vinardo_") else (0.0, 0.0))
            if delta > absolute + rel * abs(a):
                report["numeric_mismatches"].append([*key, field])
    report["ok"] = not any(value for name, value in report.items() if name != "max_errors")
    report["compared_poses"] = len(cpu[2].keys() & gpu[2].keys())
    report["full_pool_exported"] = all(integer(row["candidate_pool"]) == integer(row["exported_poses"])
        for run in (cpu, gpu) for row in run[1].values())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_dir", type=Path)
    parser.add_argument("cuda_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.cpu_dir, args.cuda_dir)
    except (ValueError, OverflowError) as error:
        print(f"Invalid export: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
