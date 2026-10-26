"""Generate a complete conformer pose pool, then rank it with global Hybrid."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from prepare_conformers import prepare, rdkit_modules


ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite docking result")
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(preparation, docking, output, allow_incomplete=False):
    metadata = read_csv(preparation / "conformers.csv")
    if not metadata or metadata[0]["molecule_id"] != "baseline":
        raise ValueError("Conformer metadata has no original baseline")
    records_path = docking / "records.csv"
    records = read_csv(records_path) if records_path.is_file() else []
    by_id = {int(row["record_index"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate docking record")
    expected = {int(row["record_index"]): row for row in metadata}
    if set(by_id) - set(expected):
        raise ValueError("Docking produced unexpected records")
    if not allow_incomplete and set(by_id) != set(expected):
        raise ValueError("Docking output is incomplete")

    # Check the full pose export, not only Top-k. RMSD is evaluation only.
    counts = {index: 0 for index in by_id}
    ids = {index: set() for index in by_id}
    best_pose = {}
    top1_from_poses = {}
    poses_path = docking / "poses.csv"
    if poses_path.is_file():
        with poses_path.open(newline="") as source:
            for row in csv.DictReader(source):
                index, pose_id = int(row["record_index"]), int(row["pose_id"])
                if index not in by_id or pose_id in ids[index]:
                    if allow_incomplete and index not in by_id:
                        continue  # A failed record can leave a partial pose export.
                    raise ValueError("Unexpected record or duplicate pose ID in pose export")
                if row["molecule_id"] != expected[index]["molecule_id"]:
                    raise ValueError("Pose ID does not belong to its conformer")
                score, rmsd = finite(row["fast_score"]), finite(row["rmsd_heavy_A"])
                if rmsd < 0:
                    raise ValueError("Negative RMSD")
                counts[index] += 1
                ids[index].add(pose_id)
                candidate = (rmsd, pose_id, score)
                if index not in best_pose or candidate[:2] < best_pose[index][:2]:
                    best_pose[index] = candidate
                fast_candidate = (-score, pose_id, rmsd)
                if index not in top1_from_poses or fast_candidate[:2] < top1_from_poses[index][:2]:
                    top1_from_poses[index] = fast_candidate
    baseline = by_id.get(int(metadata[0]["record_index"]))
    baseline_rmsd = finite(baseline["top1_rmsd_heavy_A"]) if baseline and baseline["status"] == "ok" else None
    summary = []
    for meta in metadata:
        index = int(meta["record_index"])
        row = by_id.get(index)
        result = dict(meta)
        if row is None:
            result.update(status="not_processed", delta_top1_rmsd_vs_baseline_A="", online_s="")
            summary.append(result)
            continue
        if row["molecule_id"] != meta["molecule_id"] or row["status"] not in ("ok", "no_poses", "no_triplets"):
            raise ValueError("Docking record is inconsistent with conformer metadata")
        if counts[index] != int(row["survivors"]):
            raise ValueError("Full pose export does not contain every survivor")
        if row["status"] == "ok":
            fast, top_id, rmsd = top1_from_poses[index]
            if (int(row["top1_pose_id"]) != top_id or
                    not math.isclose(finite(row["top1_fast_score"]), -fast, abs_tol=1e-10, rel_tol=1e-10) or
                    not math.isclose(finite(row["top1_rmsd_heavy_A"]), rmsd, abs_tol=1e-10, rel_tol=1e-10) or
                    not math.isclose(finite(row["best_available_rmsd_heavy_A"]), best_pose[index][0], abs_tol=1e-10, rel_tol=1e-10)):
                raise ValueError("Top-1 or oracle summary disagrees with the complete pose export")
            if int(row["success_lt_2A"]) != int(rmsd < 2.0):
                raise ValueError("Reported success disagrees with the Top-1 RMSD")
        else:
            if int(row["survivors"]) != 0:
                raise ValueError("No-pose status with survivors")
            rmsd = None
        result.update(row)
        result["delta_top1_rmsd_vs_baseline_A"] = rmsd - baseline_rmsd if rmsd is not None and baseline_rmsd is not None else ""
        result["online_s"] = sum(finite(row[key]) for key in (
            "read_parse_preparation_s", "pose_generation_s", "fast_scoring_s"))
        summary.append(result)
    fields = list(metadata[0]) + [key for key in (
        "status", "ligand_triplets", "candidate_matches", "transforms_generated", "survivors",
        "top1_pose_id", "top1_fast_score", "top1_rmsd_heavy_A", "success_lt_2A",
        "best_available_rmsd_heavy_A", "delta_top1_rmsd_vs_baseline_A", "read_parse_preparation_s",
        "pose_generation_s", "fast_scoring_s", "rmsd_evaluation_s", "online_s") if key not in metadata[0]]
    write_csv(output / "summary.csv", summary, fields)

    # Compare raw Fast values only. Never compare per-record Hybrid z-scores.
    generated = [row for row in summary[1:] if row["status"] == "ok"]
    baseline_row = summary[0]
    selected = min(generated, key=lambda row: (-finite(row["top1_fast_score"]),
                   int(row["record_index"]), int(row["top1_pose_id"]))) if generated else None
    comparison = []

    def add_comparison(label, row, oracle=False):
        entry = {"case": label, "selection_rule": "minimum RMSD; evaluation only" if oracle else
                 "maximum raw Fast; then record_index and pose_id", "status": "no_poses",
                 "record_index": "", "molecule_id": "", "pose_id": "", "fast_score": "",
                 "rmsd_heavy_A": "", "success_lt_2A": 0}
        if row is not None and row["status"] == "ok":
            index = int(row["record_index"])
            rmsd, pose_id, score = best_pose[index] if oracle else (
                finite(row["top1_rmsd_heavy_A"]), int(row["top1_pose_id"]), finite(row["top1_fast_score"]))
            entry.update(status="ok", record_index=index, molecule_id=row["molecule_id"],
                         pose_id=pose_id, fast_score=score, rmsd_heavy_A=rmsd, success_lt_2A=int(rmsd < 2.0))
        elif allow_incomplete:
            entry.update(status="incomplete_run", success_lt_2A="")
        if allow_incomplete:
            entry.update(status="incomplete_run", success_lt_2A="")
        comparison.append(entry)

    add_comparison("crystal_input_baseline", baseline_row)
    add_comparison("ensemble_selected_by_fast", selected)
    oracle = min(generated, key=lambda row: (best_pose[int(row["record_index"])][0],
                 int(row["record_index"]), best_pose[int(row["record_index"])][1])) if generated else None
    add_comparison("ensemble_best_available_rmsd", oracle, oracle=True)
    write_csv(output / "comparison.csv", comparison, list(comparison[0]))
    return {"conformer_records": len(metadata) - 1, "docked_records": len(records),
            "ensemble_online_s": sum(row["online_s"] for row in summary[1:] if row["online_s"] != ""),
            "baseline_online_s": summary[0]["online_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex", type=Path, default=Path("/workspace/datasets/refined-set/1a1e"))
    parser.add_argument("--driver", type=Path, default=ROOT / "build" / "conformer_docking")
    parser.add_argument("--hybrid-driver", type=Path, default=ROOT / "build" / "conformer_hybrid")
    parser.add_argument("--generation-only", action="store_true",
                        help="Save the complete pose pool without final scoring; it can be rescored later")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-conformers", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--min-rmsd", type=float, default=0.5)
    parser.add_argument("--energy-window", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    args = parser.parse_args()
    protocol = None
    output = args.output or ROOT / "runs" / ("1a1e-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}")
    try:
        rdkit_modules()  # Stop before generating output or opening molecules if absent.
        if output.exists():
            raise ValueError("Output directory already exists")
        driver = args.driver.resolve()
        if not driver.is_file() or not os.access(driver, os.X_OK):
            raise ValueError(f"Compile the experimental driver first: {driver}")
        hybrid_driver = args.hybrid_driver.resolve()
        if not args.generation_only and (not hybrid_driver.is_file() or not os.access(hybrid_driver, os.X_OK)):
            raise ValueError(f"Compile the Hybrid rescoring driver first: {hybrid_driver}")
        complex_path = args.complex.resolve()
        ligand = complex_path / (complex_path.name + "_ligand.mol2")
        for path in (ligand, complex_path / (complex_path.name + "_protein.pdb"),
                     complex_path / (complex_path.name + "_pocket.pdb")):
            if not path.is_file():
                raise ValueError(f"Missing input: {path}")
        output.mkdir(parents=True, exist_ok=False)
        output = output.resolve()
        started = time.perf_counter()
        protocol = {"status": "running", "platform": platform.platform(), "python": sys.version,
                    "arguments": sys.argv[1:], "complex": str(complex_path),
                    "ranking": "not performed" if args.generation_only else "global Hybrid",
                    "cutoff_A": 24, "kabsch": "quaternion", "clash": "cached",
                    "index_policy": "one shared index; no fallback",
                    "hybrid": {"lambda": 0.3, "vinardo_alpha": 0.5, "local_k": 10, "global_x": 10,
                               "normalization": "all generated-conformer survivors; baseline excluded"},
                    "fast": "generation diagnostics only; no prefilter or final pose selection",
                    "ensemble_definition": "generated conformers only; crystal original is a separate baseline",
                    "time_scope": "online excludes RMSD/export; process wall includes them and common startup"}
        protocol["driver_sha256"] = file_digest(driver)
        protocol["experiment_sources_sha256"] = {
            name: file_digest(ROOT / name) for name in (
                "prepare_conformers.py", "run_experiment.py", "conformer_docking.cpp", "CMakeLists.txt",
                "rank_hybrid.py", "conformer_hybrid.cpp", "hybrid_ranking.hpp", "hybrid_ranking.cpp")}
        protocol_path = output / "protocol.json"
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        preparation = output / "preparation"
        protocol["preparation"] = prepare(ligand, preparation, args.num_conformers, args.num_candidates,
            args.seed, args.min_rmsd, args.energy_window, args.max_iterations)
        print(f"Selected {protocol['preparation']['selected']} / {args.num_conformers} requested conformers; baseline retained separately")
        if not protocol["preparation"]["mmff_supported"]:
            print("MMFF94 unsupported: no energies were used or invented; conformers were selected by diversity")
        command = [str(driver), str(complex_path), str(preparation / "conformers.mol2"),
                   str(preparation / "reference.mol2"), str(output / "docking"), "--cutoff", "24"]
        environment = os.environ.copy()
        for key in ("DOCKING_PROFILE_STAGES", "DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS"):
            environment.pop(key, None)
        environment.update(DOCKING_KABSCH_METHOD="quaternion", DOCKING_CPU_CLASH_METHOD="cached",
                           DOCKING_QUERY_METHOD="reference", DOCKING_CPU_SCREENING_ONLY="1")
        protocol["command"] = command
        protocol["environment"] = {key: environment.get(key) for key in (
            "DOCKING_KABSCH_METHOD", "DOCKING_CPU_CLASH_METHOD", "DOCKING_QUERY_METHOD",
            "DOCKING_CPU_SCREENING_ONLY", "DOCKING_PROFILE_STAGES",
            "DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS")}
        process_start = time.perf_counter()
        with (output / "docking.stdout").open("w") as stdout, (output / "docking.stderr").open("w") as stderr:
            completed = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=False)
        protocol["docking_process_wall_s"] = time.perf_counter() - process_start
        protocol["docking_exit_status"] = completed.returncode
        protocol["summary"] = summarize(preparation, output / "docking", output, completed.returncode != 0)
        # Keep the old Fast checks as named diagnostics, never as the final ranking.
        for name in ("summary.csv", "comparison.csv"):
            (output / name).rename(output / ("fast_" + name))
        protocol["wall_through_summary_s"] = time.perf_counter() - started
        protocol["status"] = "ok" if completed.returncode == 0 else "docking_failed"
        protocol_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        if completed.returncode != 0:
            raise RuntimeError("Docking failed; partial results are marked incomplete; see docking.stderr")
        if not args.generation_only:
            command = [sys.executable, "-B", str(ROOT / "rank_hybrid.py"), "--run", str(output),
                       "--output", str(output / "hybrid"), "--driver", str(hybrid_driver),
                       "--complex", str(complex_path), "--local-k", "10", "--top-x", "10"]
            protocol["hybrid_command"] = command
            with (output / "hybrid.stdout").open("w") as stdout, (output / "hybrid.stderr").open("w") as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            protocol["hybrid_exit_status"] = result.returncode
            protocol["hybrid_output"] = str(output / "hybrid" / "ranking")
            if result.returncode != 0:
                raise RuntimeError("Hybrid rescoring failed; the full pose pool is saved for replay; see hybrid.stderr")
            print((output / "hybrid" / "ranking" / "topk_summary.csv").read_text(), end="")
        protocol["wall_total_s"] = time.perf_counter() - started
        protocol_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        print(f"Conformer experiment complete: {output}")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        if protocol is not None:
            protocol["status"] = "failed"
            protocol["error"] = str(error)
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        print(f"Conformer experiment: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
