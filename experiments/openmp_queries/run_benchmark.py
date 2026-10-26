#!/usr/bin/env python3
"""Run a fixed gpu_docking campaign and compare query-thread outputs in order."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
RANKINGS = ("fast", "weighted-interaction", "vinardo-rep50", "hybrid")
# Identity, status and integer counts stay exact even with numeric tolerances.
EXACT_COLUMNS = {
    "record_index", "molecule_id", "status", "error", "sdf", "rank", "pose_id",
    "alignment_index", "occupied_atoms", "diameter_gt_cutoff", "ligand_triplets",
    "candidate_matches", "alignments", "valid_transforms", "survivors",
    "exported_poses", "candidate_pool",
}
ENVIRONMENT_KEYS = {
    "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_NTASKS",
}
PRESENCE_ENABLED_TIMERS = ("DOCKING_STAGE_TIMINGS", "DOCKING_KABSCH_TIMINGS")


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def crop_margin(value):
    if value == "off":
        return value
    result = nonnegative_float(value)
    if result == 0:
        raise argparse.ArgumentTypeError("must be off or positive")
    return str(result)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protein", "pocket", "executable", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--ligands", type=Path,
                        default=Path("/work/rinfascelli/Commercial_higher500.mol2"))
    parser.add_argument("--limit", type=positive_integer, default=1000)
    parser.add_argument("--top-k", type=positive_integer, default=10,
                        help="exported poses per record, fixed for the campaign (default 10)")
    parser.add_argument("--threads", type=positive_integer, nargs="+", default=[1, 2, 4, 8],
                        help="query thread counts; baseline 1 is always included")
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--ranking", choices=RANKINGS, nargs="+", default=list(RANKINGS))
    parser.add_argument("--backend", choices=("cpu", "cuda"), nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--query-method", choices=("reference", "optimized"), default="reference")
    parser.add_argument("--crop-margin", type=crop_margin, default="off")
    parser.add_argument("--absolute-tolerance", type=nonnegative_float, default=0.0)
    parser.add_argument("--relative-tolerance", type=nonnegative_float, default=0.0)
    result = parser.parse_args(argv)
    result.threads = [1] + sorted(set(result.threads) - {1})
    result.ranking = list(dict.fromkeys(result.ranking))
    result.backend = list(dict.fromkeys(result.backend))
    for name in ("protein", "pocket", "ligands", "executable", "output"):
        setattr(result, name, getattr(result, name).expanduser().resolve())
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relevant_environment(environment):
    return {key: value for key, value in sorted(environment.items())
            if key.startswith(("DOCKING_", "OMP_")) or key in ENVIRONMENT_KEYS}


def run_environment(args, threads):
    environment = os.environ.copy()
    environment.update({
        "DOCKING_QUERY_THREADS": str(threads),
        "DOCKING_QUERY_METHOD": args.query_method,
        "DOCKING_KABSCH_METHOD": "quaternion",
        "DOCKING_PROFILE_STAGES": "0",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": str(threads),
    })
    # Legacy CUDA switches are enabled by presence, including the value "0".
    for key in PRESENCE_ENABLED_TIMERS:
        environment.pop(key, None)
    if args.crop_margin == "off":
        environment.pop("DOCKING_AREA_CROP_MARGIN_A", None)
    else:
        environment["DOCKING_AREA_CROP_MARGIN_A"] = args.crop_margin
    return environment


def command(args, ranking, backend, output):
    result = [str(args.executable), str(args.protein), str(args.pocket), str(args.ligands),
              str(output), "--limit", str(args.limit), "--top-k", str(args.top_k), "--cutoff", "24",
              "--kabsch", "quaternion", "--query-method", args.query_method,
              "--ranking", ranking, "--timings", "--diagnostics", "off", "--continue-on-error"]
    if ranking != "fast":
        result.extend(["--rescoring-backend", backend, "--fast-prefilter", "500"])
    if ranking in ("weighted-interaction", "hybrid"):
        result.extend(["--interaction-region", "pocket"])
    if ranking == "hybrid":
        result.extend(["--hybrid-lambda", "0.3", "--vinardo-alpha", "0.5"])
    return result


def metadata_command(argv):
    try:
        result = subprocess.run(argv, cwd=REPOSITORY, capture_output=True, text=True,
                                timeout=20, check=False)
        return {"command": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "error": str(error)}


def file_metadata(path, digest=False):
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        sha = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                sha.update(block)
        result["sha256"] = sha.hexdigest()
    return result


def campaign_metadata(args):
    options = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items()}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "options": options,
        "platform": platform.platform(), "hostname": platform.node(),
        "logical_cpus": os.cpu_count(), "python": sys.version,
        "inherited_relevant_environment": relevant_environment(os.environ),
        "inputs": {name: file_metadata(getattr(args, name))
                   for name in ("protein", "pocket", "ligands")},
        "executable": file_metadata(args.executable, digest=True),
        "runner": file_metadata(Path(__file__).resolve(), digest=True),
        "git": {name: metadata_command(["git", *flags]) for name, flags in (
            ("head", ["rev-parse", "HEAD"]), ("branch", ["branch", "--show-current"]),
            ("status", ["status", "--short"]), ("diff", ["diff", "--binary", "HEAD"]),
            ("remote", ["remote", "-v"]))},
        "nvidia_smi": metadata_command([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"]),
        "timing_note": "Per-column timing sums/means are intervals, never end-to-end wall time.",
        "comparison_note": "Ordered records.csv and poses.csv; within ranking/backend only.",
    }


def read_table(path):
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Missing or duplicate CSV header: {path.name}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path.name}")
    return fields, rows


def finite_nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Expected a finite non-negative timing")
    return result


def inspect_output(output, returncode, limit, wall, requested_threads=1):
    result = {"status": "incomplete", "returncode": returncode,
              "requested_records": limit, "completed_records": 0,
              "successful_records": 0, "failed_records": 0, "records_with_poses": 0,
              "external_wall_s": wall, "successful_records_per_s": None,
              "completed_records_per_s": None, "startup_s": None,
              "wall_before_summary_s": None, "online_wall_s": None,
              "successful_online_records_per_s": None,
              "timings": {}, "query_thread_use": {},
              "issues": []}
    records = []
    try:
        _, records = read_table(output / "records.csv")
        _, poses = read_table(output / "poses.csv")
        result["completed_records"] = len(records)
        result["failed_records"] = sum(row["status"] == "error" for row in records)
        result["successful_records"] = len(records) - result["failed_records"]
        result["records_with_poses"] = sum(row["status"] == "ok" for row in records)
        if [int(row["record_index"]) for row in records] != list(range(1, len(records) + 1)):
            raise ValueError("Record indices are not the original contiguous order")
        if any(row["status"] not in ("ok", "no_poses", "no_triplets", "error") for row in records):
            raise ValueError("Unknown record status")
        by_index = {row["record_index"]: row for row in records}
        counts = Counter()
        previous = 0
        for row in poses:
            index = row["record_index"]
            if index not in by_index or row["molecule_id"] != by_index[index]["molecule_id"]:
                raise ValueError("Pose does not belong to its record")
            counts[index] += 1
            if int(index) < previous or int(row["rank"]) != counts[index]:
                raise ValueError("Pose order/ranks are inconsistent")
            previous = int(index)
        for row in records:
            if counts[row["record_index"]] != int(row["exported_poses"]):
                raise ValueError("Pose count does not match records.csv")
            if row["status"] == "error" and counts[row["record_index"]]:
                raise ValueError("An error record has exported poses")
        fields, timings = read_table(output / "timings.csv")
        successful = [row for row in records if row["status"] != "error"]
        if [row["record_index"] for row in timings] != [row["record_index"] for row in successful]:
            raise ValueError("Timing rows do not preserve successful record indices")
        for field in fields:
            if not field.endswith("_s"):
                continue
            values = [finite_nonnegative(row[field]) for row in timings if row[field] != ""]
            result["timings"][field] = {"sum_s": sum(values),
                "mean_s": statistics.mean(values) if values else None, "count": len(values)}
        _, summary_rows = read_table(output / "summary.csv")
        summary = {row["key"]: row["value"] for row in summary_rows}
        result["engine_status"] = summary["status"]
        result["query_openmp_available"] = summary["query_openmp_available"] in ("1", "true")
        result["query_parallel_min_triplets"] = int(summary["query_parallel_min_triplets"])
        if int(summary["query_threads_requested"]) != requested_threads:
            raise ValueError("Summary query thread request differs from the command environment")
        used = Counter(int(row["query_threads_used"]) for row in timings)
        if any(int(row["query_threads_requested"]) != requested_threads for row in timings):
            raise ValueError("Timing query thread request differs from the command environment")
        if any(threads < 1 or threads > requested_threads for threads in used):
            raise ValueError("Invalid actual query thread count")
        result["query_thread_use"] = {
            "counts": dict(sorted(used.items())),
            "parallel_records": sum(count for threads, count in used.items() if threads > 1),
            "serial_records": used.get(1, 0),
            "all_serial": bool(timings) and all(threads == 1 for threads in used),
        }
        result["startup_s"] = finite_nonnegative(summary["startup_s"])
        result["wall_before_summary_s"] = finite_nonnegative(summary["wall_before_summary_s"])
        if int(summary["processed"]) != len(records):
            raise ValueError("Summary processed count disagrees with records.csv")
        for key in ("successful_records", "failed_records"):
            if int(summary[key]) != result[key]:
                raise ValueError(f"Summary {key} count disagrees with records.csv")
        expected_status = "complete_with_errors" if result["failed_records"] else "complete"
        expected_code = 1 if result["failed_records"] else 0
        if summary["status"] != expected_status or returncode != expected_code:
            raise ValueError("Exit code or summary status does not match completed record errors")
        if len(records) != limit:
            raise ValueError(f"Requested {limit} records, obtained {len(records)} (possibly short input)")
        online_wall = result["wall_before_summary_s"] - result["startup_s"]
        if online_wall < 0:
            raise ValueError("Startup interval exceeds engine wall time")
        result["status"] = expected_status
        result["online_wall_s"] = online_wall
        if online_wall > 0:
            result["successful_online_records_per_s"] = result["successful_records"] / online_wall
        if wall > 0:
            result["successful_records_per_s"] = result["successful_records"] / wall
            result["completed_records_per_s"] = len(records) / wall
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["issues"].append(str(error))
    return result


def compare_tables(baseline, candidate, absolute_tolerance=0.0, relative_tolerance=0.0):
    """Compare every exported field in order; never rematch or sort records/poses."""
    differences = []
    for filename in ("records.csv", "poses.csv"):
        try:
            fields, reference = read_table(baseline / filename)
            other_fields, other = read_table(candidate / filename)
            if fields != other_fields:
                differences.append(f"{filename}: column order/header differs")
                continue
            if len(reference) != len(other):
                differences.append(f"{filename}: row count {len(reference)} != {len(other)}")
            for number, (left, right) in enumerate(zip(reference, other), start=2):
                for field in fields:
                    if left[field] == right[field]:
                        continue
                    close = False
                    if field not in EXACT_COLUMNS and (absolute_tolerance or relative_tolerance):
                        try:
                            a, b = float(left[field]), float(right[field])
                            close = math.isfinite(a) and math.isfinite(b) and math.isclose(
                                a, b, abs_tol=absolute_tolerance, rel_tol=relative_tolerance)
                        except ValueError:
                            pass
                    if not close:
                        differences.append(f"{filename}:{number}:{field}: {left[field]!r} != {right[field]!r}")
                        if len(differences) >= 20:
                            return {"equal": False, "differences": differences, "truncated": True}
        except (OSError, ValueError) as error:
            differences.append(f"{filename}: {error}")
    return {"equal": not differences, "differences": differences, "truncated": False}


def execute_run(args, ranking, backend, threads, repeat):
    directory = args.output / ranking / backend / f"threads-{threads}" / f"repeat-{repeat}"
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "docking"  # gpu_docking creates this new directory itself.
    argv = command(args, ranking, backend, output)
    environment = run_environment(args, threads)
    write_json(directory / "command.json", {"argv": argv, "cwd": str(REPOSITORY),
        "relevant_environment": relevant_environment(environment),
        "unset_environment": list(PRESENCE_ENABLED_TIMERS) + (
            ["DOCKING_AREA_CROP_MARGIN_A"] if args.crop_margin == "off" else [])})
    start = time.perf_counter()
    returncode = None
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        try:
            completed = subprocess.run(argv, cwd=REPOSITORY, env=environment,
                                       stdout=stdout, stderr=stderr, check=False)
            returncode = completed.returncode
        except OSError as error:
            stderr.write(str(error) + "\n")
    result = inspect_output(output, returncode, args.limit, time.perf_counter() - start, threads)
    if (result["status"] != "incomplete" and max(args.threads) > 1
            and not result["query_openmp_available"]):
        result["status"] = "incomplete"
        result["successful_records_per_s"] = result["completed_records_per_s"] = None
        result["successful_online_records_per_s"] = None
        result["issues"].append("Parallel campaign requires a build with ENABLE_OPENMP=ON")
    result.update(ranking=ranking, backend=backend, threads=threads, repeat=repeat,
                  directory=str(directory.relative_to(args.output)))
    write_json(directory / "result.json", result)
    return result


def aggregate(results):
    groups = {}
    for result in results:
        key = (result["ranking"], result["backend"], result["threads"])
        groups.setdefault(key, []).append(result)
    rows = []
    for (ranking, backend, threads), runs in groups.items():
        complete = [run for run in runs if run["status"] in ("complete", "complete_with_errors")]
        row = {"ranking": ranking, "backend": backend, "threads": threads,
               "repeats": len(runs), "completed_repeats": len(complete),
               "all_comparisons_equal": all(run["comparison"]["equal"] for run in runs)}
        row["actual_query_threads_used"] = sorted({int(used) for run in complete
            for used in run["query_thread_use"]["counts"]})
        use_counts = Counter()
        for run in complete:
            for used, count in run["query_thread_use"]["counts"].items():
                use_counts[int(used)] += count
        row["actual_query_thread_use_counts"] = dict(sorted(use_counts.items()))
        row["all_records_used_serial_queries"] = bool(complete) and all(
            run["query_thread_use"]["all_serial"] for run in complete)
        for field in ("external_wall_s", "startup_s", "wall_before_summary_s",
                      "online_wall_s", "successful_online_records_per_s",
                      "successful_records", "failed_records", "successful_records_per_s",
                      "completed_records_per_s"):
            values = [run[field] for run in complete if run[field] is not None]
            row[field + "_mean"] = statistics.mean(values) if values else None
        row["timing_column_sum_s_mean"] = {
            field: statistics.mean(run["timings"][field]["sum_s"] for run in complete)
            for field in complete[0]["timings"]} if complete else {}
        rows.append(row)
    baselines = {(row["ranking"], row["backend"]): row for row in rows if row["threads"] == 1}
    for row in rows:
        baseline = baselines[(row["ranking"], row["backend"])]
        wall, baseline_wall = row["external_wall_s_mean"], baseline["external_wall_s_mean"]
        # Incomplete/mismatching groups must not advertise a speedup.
        valid = (row["completed_repeats"] == row["repeats"] and row["all_comparisons_equal"]
                 and baseline["completed_repeats"] == baseline["repeats"]
                 and baseline["all_comparisons_equal"])
        row["external_wall_speedup_vs_threads_1"] = baseline_wall / wall if valid and wall else None
        for interval in ("search_completed_s", "query_preparation_s", "gpu_search_s"):
            duration = row["timing_column_sum_s_mean"].get(interval)
            baseline_duration = baseline["timing_column_sum_s_mean"].get(interval)
            row[interval.removesuffix("_s") + "_interval_speedup_vs_threads_1"] = (
                baseline_duration / duration if valid and duration and baseline_duration is not None else None)
    return rows


def run_campaign(args):
    for name in ("protein", "pocket", "ligands", "executable"):
        if not getattr(args, name).is_file():
            raise ValueError(f"Missing {name}: {getattr(args, name)}")
    if not os.access(args.executable, os.X_OK):
        raise ValueError(f"Executable is not executable: {args.executable}")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "campaign.json", campaign_metadata(args))
    results = []
    for ranking in args.ranking:
        for backend in (["not-applicable"] if ranking == "fast" else args.backend):
            baseline = None
            for threads in args.threads:
                for repeat in range(1, args.repeats + 1):
                    print(f"{ranking} backend={backend} threads={threads} repeat={repeat}", flush=True)
                    result = execute_run(args, ranking, backend, threads, repeat)
                    if threads > 1 and result["query_thread_use"].get("all_serial"):
                        print("All successful records used the serial query fallback in this run.", flush=True)
                    output = args.output / result["directory"] / "docking"
                    if baseline is None:
                        baseline = output
                    result["baseline_directory"] = str(baseline.relative_to(args.output))
                    result["comparison"] = compare_tables(baseline, output,
                        args.absolute_tolerance, args.relative_tolerance)
                    write_json(output.parent / "result.json", result)
                    results.append(result)
                    write_json(args.output / "runs.json", results)
                    write_json(args.output / "summary.json", aggregate(results))
                    if result["status"] == "incomplete":
                        print("Incomplete run; campaign stopped. See result.json and stderr.log.", file=sys.stderr)
                        return 1
    equal = all(run["comparison"]["equal"] for run in results)
    print(f"Completed {len(results)} runs; comparisons {'match' if equal else 'differ'}; "
          f"results: {args.output}", flush=True)
    return 0 if equal else 1


def main(argv=None):
    try:
        return run_campaign(arguments(argv))
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
