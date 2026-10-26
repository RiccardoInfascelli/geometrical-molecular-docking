"""Rescore a complete saved conformer pool, without rerunning geometric search."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
POSE_FIELDS = [
    "record_index", "pose_id", "fast_score", "clash_fraction", "occupied_atoms",
    "rmsd_heavy_A", "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
    "tx", "ty", "tz",
]


def rows(path):
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or duplicate CSV header: {path}")
        result = list(reader)
        if any(None in row or None in row.values() for row in result):
            raise ValueError(f"Malformed CSV row: {path}")
        return result


def integer(value, label, minimum=0):
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"Invalid integer {label}: {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"Invalid {label}: {value!r}")
    return result


def finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}")
    return result


def write_pose_input(run, tsv_path):
    """Validate every saved survivor and export all numeric fields in source order."""
    run, tsv_path = Path(run), Path(tsv_path)
    with (run / "protocol.json").open() as source:
        source_protocol = json.load(source)
    # A later analysis failure must not invalidate a complete, successful search.
    if type(source_protocol.get("docking_exit_status")) is not int or source_protocol["docking_exit_status"] != 0:
        raise ValueError("Source docking did not exit successfully")
    source_complex = source_protocol.get("complex")
    if not isinstance(source_complex, str) or not source_complex:
        raise ValueError("Source protocol does not identify the complex")
    preparation, docking = run / "preparation", run / "docking"
    for name in ("conformers.mol2", "reference.mol2"):
        if not (preparation / name).is_file():
            raise ValueError(f"Missing prepared input: {name}")
    metadata = rows(preparation / "conformers.csv")
    if (len(metadata) < 2 or metadata[0].get("record_index") != "1" or
            metadata[0].get("molecule_id") != "baseline" or
            metadata[0].get("source") != "crystal_original"):
        raise ValueError("First conformer must be the separate crystal_original baseline, record 1")
    expected, names = {}, set()
    for position, row in enumerate(metadata, 1):
        index = integer(row["record_index"], "conformer record_index", 1)
        name = row["molecule_id"]
        if index != position or not name or name in names:
            raise ValueError("Duplicate or out-of-order conformer identity")
        if index > 1 and (name == "baseline" or row["source"] != "etkdgv3"):
            raise ValueError("Generated ensemble must remain distinct from the crystal baseline")
        expected[index] = name
        names.add(name)

    protocol_rows = rows(docking / "protocol.csv")
    protocol = {row["key"]: row["value"] for row in protocol_rows}
    if len(protocol) != len(protocol_rows):
        raise ValueError("Duplicate source docking protocol key")
    if protocol.get("pose_pool") != "all operational clash survivors; no prefilter or deduplication":
        raise ValueError("Source pool is not the complete set of clash survivors")
    if not protocol.get("index_policy", "").startswith("shared_index_only;"):
        raise ValueError("Source docking did not use the shared-only index")
    if integer(protocol.get("record_count", ""), "protocol record_count", 1) != len(expected):
        raise ValueError("Source protocol has an incomplete record count")

    records, counts = {}, {}
    for row in rows(docking / "records.csv"):
        index = integer(row["record_index"], "docking record_index", 1)
        if index not in expected or index in records or row["molecule_id"] != expected[index]:
            raise ValueError("Unexpected or duplicate docking record")
        count = integer(row["survivors"], "survivors")
        if row["status"] not in ("ok", "no_poses", "no_triplets"):
            raise ValueError("Source record has an unsuccessful status")
        if (row["status"] == "ok") != (count > 0):
            raise ValueError("Source status disagrees with the survivor count")
        records[index], counts[index] = row, count
    if set(records) != set(expected):
        raise ValueError("Source docking records are incomplete")

    seen = {index: set() for index in expected}
    with (docking / "poses.csv").open(newline="") as source, tsv_path.open("x", newline="") as target:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not set(POSE_FIELDS + ["molecule_id"]).issubset(reader.fieldnames):
            raise ValueError("Source pose CSV is missing required fields")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Source pose CSV has duplicate fields")
        writer = csv.DictWriter(target, fieldnames=POSE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if None in row or None in row.values():
                raise ValueError("Malformed saved pose CSV")
            index = integer(row["record_index"], "pose record_index", 1)
            pose_id = integer(row["pose_id"], "pose_id")
            if index not in expected or row["molecule_id"] != expected[index]:
                raise ValueError("Pose does not belong to a declared conformer")
            if pose_id in seen[index] or pose_id >= counts[index]:
                raise ValueError("Duplicate or out-of-range pose identity")
            integer(row["occupied_atoms"], "occupied_atoms")
            for field in POSE_FIELDS[2:]:
                number = finite(row[field], field)
                if field == "rmsd_heavy_A" and number < 0:
                    raise ValueError("Negative saved RMSD")
                if field == "clash_fraction" and not 0 <= number <= 1:
                    raise ValueError("Invalid saved clash fraction")
            seen[index].add(pose_id)
            writer.writerow({field: row[field] for field in POSE_FIELDS})
    if any(len(seen[index]) != counts[index] for index in expected):
        raise ValueError("Saved pose export is truncated; every survivor is required")
    return {"source_complex": source_complex, "record_count": len(expected),
            "conformer_count": len(expected) - 1, "baseline_poses": counts[1],
            "ensemble_poses": sum(count for index, count in counts.items() if index != 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--driver", type=Path, default=ROOT / "build" / "conformer_hybrid")
    parser.add_argument("--complex", type=Path)
    parser.add_argument("--interaction-region", choices=("pocket", "protein"), default="pocket")
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument("--top-x", type=int, default=10)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "runs" / (
        "1a1e-hybrid-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}")
    protocol = None
    started = time.perf_counter()
    try:
        if args.top_x < 10 or args.local_k < args.top_x:
            raise ValueError("Require local-k >= top-x >= 10 to evaluate Top-1/5/10")
        if not args.driver.is_file():
            raise ValueError(f"Hybrid rescoring executable is missing: {args.driver}")
        output.mkdir(parents=True, exist_ok=False)
        protocol = {"status": "validating", "source_run": str(args.run.resolve()),
                    "driver": str(args.driver.resolve()), "interaction_region": args.interaction_region,
                    "local_k": args.local_k, "top_x": args.top_x, "hybrid_lambda": 0.3,
                    "vinardo_repulsion_alpha": 0.5, "geometric_search_rerun": False,
                    "pose_selection_before_rescoring": "none; all original survivors",
                    "baseline": "separate crystal-original record; excluded from generated ensemble",
                    "source_outputs_modified": False}
        metadata = write_pose_input(args.run, output / "pose_input.tsv")
        protocol.update(metadata)
        complex_path = str(args.complex) if args.complex else metadata["source_complex"]
        preparation = args.run.resolve() / "preparation"
        command = [str(args.driver.resolve()), complex_path, str(preparation / "conformers.mol2"),
                   str(preparation / "reference.mol2"), str((output / "pose_input.tsv").resolve()),
                   str((output / "ranking").resolve()), "--interaction-region", args.interaction_region,
                   "--local-k", str(args.local_k), "--top-x", str(args.top_x)]
        protocol.update(command=command, status="running")
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        protocol["rescoring_exit_status"] = result.returncode
        if result.returncode != 0:
            raise RuntimeError(f"Hybrid rescoring failed with exit status {result.returncode}; see {output / 'stderr.log'}")
        summary = output / "ranking" / "topk_summary.csv"
        if not summary.is_file():
            raise RuntimeError("Rescoring returned success without topk_summary.csv")
        protocol["status"] = "ok"
        print(summary.read_text(), end="")
        print(f"Hybrid analysis saved in {output}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if protocol is not None:
            protocol.update(status="failed", error=str(error))
        print(f"rank_hybrid: {error}", file=sys.stderr)
        return 1
    finally:
        if protocol is not None:
            protocol["wall_s"] = time.perf_counter() - started
            with (output / "protocol.json").open("w") as target:
                json.dump(protocol, target, indent=2)
                target.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
