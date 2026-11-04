#!/usr/bin/env python3
"""Compare sequential and bounded CPU-query/CUDA overlap on the same records."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
HELPER_PATH = REPOSITORY / "experiments/openmp_queries/run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("overlap_existing_benchmark", HELPER_PATH)
EXISTING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXISTING)
RANKINGS = ("fast", "weighted-interaction", "vinardo", "vinardo-rep50", "hybrid")


def prefilter(value):
    return "none" if value == "none" else str(EXISTING.positive_integer(value))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex", type=Path,
                        default=Path("/work/rinfascelli/tesi/datasets/refined-set/1a1e"))
    parser.add_argument("--ligands", type=Path,
                        default=Path("/work/rinfascelli/Commercial_higher500.mol2"))
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cmake-cache", type=Path)
    parser.add_argument("--limit", type=EXISTING.positive_integer)
    parser.add_argument("--threads", type=int, nargs="+", choices=(1, 4, 8), default=[1, 8])
    parser.add_argument("--overlap", nargs="+", choices=("off", "on"), default=["off", "on"])
    parser.add_argument("--ranking", choices=RANKINGS, default="fast")
    parser.add_argument("--backend", choices=("cuda",), default="cuda",
                        help="rescoring backend; non-Fast overlap supports CUDA only")
    parser.add_argument("--fast-prefilter", type=prefilter, default="none",
                        help="non-Fast rescoring pool: none (default) or a positive Fast top-N")
    parser.add_argument("--vinardo-alpha", type=EXISTING.nonnegative_float)
    parser.add_argument("--hybrid-lambda", type=EXISTING.nonnegative_float)
    parser.add_argument("--repeats", type=EXISTING.positive_integer, default=3)
    parser.add_argument("--top-k", type=EXISTING.positive_integer, default=10)
    parser.add_argument("--smoke-only", action="store_true",
                        help="one repetition, at most three records, every pose in the declared ranking pool exported")
    parser.add_argument("--crop-margin", type=EXISTING.crop_margin, default="off")
    parser.add_argument("--absolute-tolerance", type=EXISTING.nonnegative_float, default=0.0)
    parser.add_argument("--relative-tolerance", type=EXISTING.nonnegative_float, default=0.0)
    parser.add_argument("--timeout-s", type=EXISTING.positive_integer, default=3600)
    args = parser.parse_args(argv)
    args.threads = sorted(set(args.threads))
    args.overlap = list(dict.fromkeys(args.overlap))
    if args.ranking == "fast" and args.fast_prefilter != "none":
        parser.error("--fast-prefilter requires a rescoring ranking")
    if args.vinardo_alpha is not None and args.ranking not in ("vinardo", "vinardo-rep50", "hybrid"):
        parser.error("--vinardo-alpha requires Vinardo or Hybrid")
    if args.ranking == "vinardo-rep50" and args.vinardo_alpha not in (None, 0.5):
        parser.error("vinardo-rep50 fixes --vinardo-alpha to 0.5")
    if args.hybrid_lambda is not None and (args.ranking != "hybrid" or args.hybrid_lambda > 1):
        parser.error("--hybrid-lambda requires Hybrid and a value in [0, 1]")
    args.limit = args.limit if args.limit is not None else (1 if args.smoke_only else 1000)
    if args.smoke_only:
        if args.limit > 3:
            parser.error("Full-pool smoke accepts at most three records")
        args.repeats = 1
    for name in ("complex", "ligands", "executable", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    identifier = args.complex.name
    args.protein = args.complex / f"{identifier}_protein.pdb"
    args.pocket = args.complex / f"{identifier}_pocket.pdb"
    args.cmake_cache = (args.cmake_cache or args.executable.parent / "CMakeCache.txt").expanduser().resolve()
    args.query_method = "reference"
    return args


def save_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def repetition_scope(args):
    if args.smoke_only:
        return "correctness_smoke"
    return "single_repeat_exploratory" if args.repeats == 1 else "repeated_measurement"


def scientific_protocol(args):
    vinardo = args.ranking in ("vinardo", "vinardo-rep50", "hybrid")
    alpha = args.vinardo_alpha if args.vinardo_alpha is not None else (0.5 if args.ranking == "vinardo-rep50" else 1.0)
    return {
        "ranking_requested": args.ranking,
        "ranking_method": "vinardo" if args.ranking == "vinardo-rep50" else args.ranking,
        "rescoring_backend": args.backend if args.ranking != "fast" else "not_calculated",
        "fast_prefilter": args.fast_prefilter,
        "vinardo_alpha": alpha if vinardo else None,
        "hybrid_lambda": (args.hybrid_lambda if args.hybrid_lambda is not None else 0.3)
            if args.ranking == "hybrid" else None,
        "kabsch_method": "quaternion", "query_method": "reference",
        "index_policy": "shared_index_only", "index_cutoff_A": 24,
        "interaction_region": "pocket" if args.ranking in ("weighted-interaction", "hybrid") else "not_calculated",
        "chemical_area_crop_margin_A": args.crop_margin,
        "target_clearance_enabled": False,
    }


def build_metadata(args):
    if args.cmake_cache.parent != args.executable.parent:
        raise ValueError("CMake cache must belong to the executable's build directory")
    cache = {}
    for line in args.cmake_cache.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line and ":" in line.split("=", 1)[0]:
            key_type, value = line.split("=", 1)
            cache[key_type.split(":", 1)[0]] = value
    for key, expected in (("CMAKE_BUILD_TYPE", "Release"), ("ENABLE_CUDA", "ON"), ("ENABLE_OPENMP", "ON")):
        if cache.get(key) != expected:
            raise ValueError(f"Require {key}={expected}, got {cache.get(key)!r}")
    scientific_sources = {}
    for folder in ("src", "include", "apps", "experiments/cpu_gpu_overlap", "experiments/openmp_queries"):
        for path in sorted((REPOSITORY / folder).rglob("*")):
            if path.is_file() and path.suffix in (".cpp", ".cu", ".hpp", ".py"):
                scientific_sources[str(path.relative_to(REPOSITORY))] = EXISTING.file_metadata(path, digest=True)
    for name in ("CMakeLists.txt", "CMakePresets.json"):
        scientific_sources[name] = EXISTING.file_metadata(REPOSITORY / name, digest=True)
    saved_cache = {k: v for k, v in cache.items() if k.startswith(("CMAKE_", "ENABLE_", "BUILD_", "TARGET_CPU", "OpenMP"))}
    return {
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scientific_protocol": scientific_protocol(args),
        "repetition_scope": repetition_scope(args),
        "repetition_note": "A single repetition does not estimate dispersion or establish statistical robustness.",
        "inputs": {k: EXISTING.file_metadata(getattr(args, k)) for k in ("protein", "pocket", "ligands")},
        "executable": EXISTING.file_metadata(args.executable, digest=True),
        "cmake_cache": {"identity": EXISTING.file_metadata(args.cmake_cache, digest=True), "variables": saved_cache},
        "source_files": scientific_sources,
        "git": {key: EXISTING.metadata_command(["git", *flags]) for key, flags in (
            ("head", ["rev-parse", "HEAD"]), ("branch", ["branch", "--show-current"]),
            ("status", ["status", "--short"]), ("diff", ["diff", "--binary", "HEAD"]))},
        "cmake_version": EXISTING.metadata_command(["cmake", "--version"]),
        "compiler_version": EXISTING.metadata_command([cache["CMAKE_CXX_COMPILER"], "--version"])
            if cache.get("CMAKE_CXX_COMPILER") else None,
        "nvidia_smi": EXISTING.metadata_command([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"]),
        "inherited_environment": EXISTING.relevant_environment(os.environ),
        "experiment": "one ranking per campaign; CUDA rescoring when requested; same retained buffer workspace and one GPU consumer",
        "timing_note": "online_wall_s is directly measured; per-record work intervals overlap and are not additive",
    }


def check_resources(args):
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("Run the CUDA campaign on an allocated Linux machine")
    physical = set()
    for number in os.sched_getaffinity(0):
        root = Path(f"/sys/devices/system/cpu/cpu{number}/topology")
        physical.add(((root / "physical_package_id").read_text().strip(), (root / "core_id").read_text().strip()))
    required = max(args.threads) + 1
    if len(physical) < required:
        raise ValueError(f"Allocate at least {required} physical CPU cores: query team plus the single GPU consumer")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and int(allocated) < required:
        raise ValueError(f"SLURM_CPUS_PER_TASK must be at least {required}")
    for key in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
        if os.environ.get(key):
            raise ValueError("Disable CUDA MPS for this separate overlap experiment")
    gpu = EXISTING.metadata_command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if gpu.get("returncode") != 0 or gpu.get("stdout", "").strip():
        raise ValueError("Cannot verify an idle GPU, or another GPU compute process is running")
    return {"available_physical_cores": len(physical), "required_cores": required,
            "affinity": sorted(os.sched_getaffinity(0)), "gpu_idle_check": gpu}


def run_environment(args, threads):
    environment = EXISTING.run_environment(args, threads)
    environment.pop("OMP_PLACES", None)
    environment.update(OMP_THREAD_LIMIT=str(threads), OMP_MAX_ACTIVE_LEVELS="1", OMP_PROC_BIND="FALSE",
                       OMP_WAIT_POLICY="PASSIVE",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    return environment


def command(args, overlap, output):
    result = EXISTING.command(args, "fast", "not-applicable", output)
    result[result.index("--ranking") + 1] = args.ranking
    if args.ranking != "fast":
        result += ["--rescoring-backend", args.backend, "--fast-prefilter", args.fast_prefilter]
    if args.ranking in ("weighted-interaction", "hybrid"):
        result += ["--interaction-region", "pocket"]
    protocol = scientific_protocol(args)
    if protocol["vinardo_alpha"] is not None:
        result += ["--vinardo-alpha", str(protocol["vinardo_alpha"])]
    if protocol["hybrid_lambda"] is not None:
        result += ["--hybrid-lambda", str(protocol["hybrid_lambda"])]
    if overlap != "legacy":
        result += ["--cpu-gpu-overlap", overlap]
    result += ["--pose-output", "all" if args.smoke_only else "topk", "--pose-detail", "full"]
    return result


def configurations(args):
    pairs = [(thread, mode) for thread in args.threads for mode in args.overlap]
    for repeat in range(args.repeats):
        shift = repeat % len(pairs)
        for thread, mode in pairs[shift:] + pairs[:shift]:
            yield repeat + 1, thread, mode


def observed_overlap(timeline, phase="search"):
    query, search = [], []
    for row in timeline:
        record = int(row["record_index"])
        for collection, start, stop in ((query, "query_start_s", "query_end_s"),
                                        (search, phase + "_start_s", phase + "_end_s")):
            if bool(row[start]) != bool(row[stop]):
                raise ValueError("Incomplete timeline interval")
            if row[start]:
                begin = EXISTING.finite_nonnegative(row[start])
                end = EXISTING.finite_nonnegative(row[stop])
                if end < begin:
                    raise ValueError("Timeline interval ends before it starts")
                if end > begin:
                    collection.append((begin, end, record))
    query.sort()
    search.sort()
    total, overlapping_queries = 0.0, set()
    cursor = 0
    for begin, end, record in query:
        while cursor < len(search) and search[cursor][1] <= begin:
            cursor += 1
        position = cursor
        while position < len(search) and search[position][0] < end:
            other_start, other_end, other_record = search[position]
            overlap = min(end, other_end) - max(begin, other_start)
            if overlap > 0:
                if other_record == record:
                    raise ValueError("A record's search began before its queries were ready")
                total += overlap
                overlapping_queries.add(record)
            position += 1
    return total, len(overlapping_queries)


def check_protocol(args, summary):
    protocol = scientific_protocol(args)
    if summary["ranking_method"] != protocol["ranking_method"] or summary["sdf_export"] not in ("0", "false"):
        raise ValueError("Unexpected scoring method or SDF export")
    for key, expected in (("ranking_requested", args.ranking), ("kabsch_method_effective", "quaternion"),
                          ("query_method_effective", "reference"), ("index_policy", "shared_index_only"),
                          ("fast_prefilter", args.fast_prefilter), ("min_target_clearance_A", "off"),
                          ("diagnostics_effective", "off"),
                          ("pose_output", "all" if args.smoke_only else "topk"), ("pose_detail", "full")):
        if summary[key] != expected:
            raise ValueError(f"Unexpected {key}: expected {expected!r}, got {summary[key]!r}")
    if float(summary["index_cutoff_A"]) != 24:
        raise ValueError("Unexpected protein index cutoff")
    if summary["target_clearance_enabled"] not in ("0", "false"):
        raise ValueError("Unexpected target clearance pruning")
    if summary["timings_enabled"] not in ("1", "true"):
        raise ValueError("Completed timing collection was not enabled")
    crop = summary["chemical_area_crop_margin_A"]
    if (args.crop_margin == "off" and crop != "off") or (
            args.crop_margin != "off" and float(crop) != float(args.crop_margin)):
        raise ValueError("Unexpected chemical-area CROP margin")
    if args.ranking != "fast":
        if summary["rescoring_backend"] != "cuda" or summary["fast_prefilter"] != args.fast_prefilter:
            raise ValueError("Unexpected rescoring backend or Fast prefilter")
    if args.ranking in ("weighted-interaction", "hybrid") and summary["interaction_region"] != "pocket":
        raise ValueError("Unexpected WeightedInteraction region")
    if args.ranking in ("vinardo", "vinardo-rep50", "hybrid"):
        if float(summary["vinardo_alpha"]) != protocol["vinardo_alpha"]:
            raise ValueError("Unexpected Vinardo alpha")
    if args.ranking == "hybrid":
        if float(summary["hybrid_lambda"]) != protocol["hybrid_lambda"]:
            raise ValueError("Unexpected Hybrid lambda")


def check_full_pool(args, records, result):
    if not result["records_with_poses"]:
        raise ValueError("Full-pool smoke did not generate valid poses")
    pool = "survivors" if args.ranking == "fast" else "candidate_pool"
    if any(int(r[pool]) != int(r["exported_poses"]) for r in records if r["status"] != "error"):
        raise ValueError("Full-pool smoke did not export the complete declared ranking pool")


def check_extended_intervals(timeline):
    for row in timeline:
        values = {}
        for phase in ("dock", "search", "rescoring"):
            begin, end = row[phase + "_start_s"], row[phase + "_end_s"]
            if bool(begin) != bool(end):
                raise ValueError("Incomplete extended timeline interval")
            if begin:
                values[phase] = (EXISTING.finite_nonnegative(begin), EXISTING.finite_nonnegative(end))
                if values[phase][1] < values[phase][0]:
                    raise ValueError("Extended timeline interval ends before it starts")
        if not values and row["status"] != "error":
            raise ValueError("Missing completed docking/search/rescoring interval")
        if values:
            if row["status"] == "error" and set(values) == {"dock"}:
                continue  # Failed calls cannot publish completed inner-stage spans.
            if set(values) != {"dock", "search", "rescoring"}:
                raise ValueError("Missing completed docking/search/rescoring interval")
            dock, search, rescoring = (values[p] for p in ("dock", "search", "rescoring"))
            if not (dock[0] <= search[0] <= search[1] <= rescoring[0] <= rescoring[1] <= dock[1]):
                raise ValueError("Extended timeline stages violate docking dependencies")


def inspect_output(args, output, returncode, wall, threads, overlap):
    result = EXISTING.inspect_output(output, returncode, args.limit, wall, threads)
    result.update(overlap=overlap, threads=threads, host_overlap_observed=False)
    if result["status"] == "incomplete":
        return result
    try:
        _, summary_rows = EXISTING.read_table(output / "summary.csv")
        summary = {r["key"]: r["value"] for r in summary_rows}
        check_protocol(args, summary)
        if args.smoke_only and overlap == "legacy":
            _, records = EXISTING.read_table(output / "records.csv")
            check_full_pool(args, records, result)
            # Legacy collection lacks the new direct online boundary. Its role
            # here is scientific equivalence, never a performance baseline.
            result["online_wall_s"] = result["successful_online_records_per_s"] = None
            return result
        if summary["cpu_gpu_overlap"] != overlap or int(summary["queue_capacity"]) != 1:
            raise ValueError("Executable did not apply the requested bounded overlap mode")
        if summary["overlap_trace"] not in ("0", "false"):
            raise ValueError("Performance and correctness runs must not enable diagnostic NVTX tracing")
        result["nvtx_available"] = summary["nvtx_available"] in ("1", "true")
        if not result["query_openmp_available"]:
            raise ValueError("This comparison requires ENABLE_OPENMP=ON")
        result["online_wall_s"] = EXISTING.finite_nonnegative(summary["online_wall_s"])
        if result["online_wall_s"] <= 0 or result["online_wall_s"] > result["wall_before_summary_s"]:
            raise ValueError("Invalid directly measured online wall interval")
        result["successful_online_records_per_s"] = result["successful_records"] / result["online_wall_s"]
        result["completed_online_records_per_s"] = result["completed_records"] / result["online_wall_s"]
        for key in ("consumer_wait_s", "producer_queue_wait_s", "query_search_host_overlap_s"):
            result[key] = EXISTING.finite_nonnegative(summary[key])
        result["query_search_overlap_records"] = int(summary["query_search_overlap_records"])
        _, timeline = EXISTING.read_table(output / "overlap_timeline.csv")
        _, records = EXISTING.read_table(output / "records.csv")
        if [r["record_index"] for r in timeline] != [r["record_index"] for r in records]:
            raise ValueError("Timeline does not preserve all original record indices")
        if any(a["status"] != b["status"] for a, b in zip(timeline, records)):
            raise ValueError("Timeline statuses do not match scientific output")
        phases = ("search",) if args.ranking == "fast" else ("search", "rescoring", "dock")
        if args.ranking != "fast":
            check_extended_intervals(timeline)
        for phase in phases:
            interval, count = observed_overlap(timeline, phase)
            duration_key, count_key = f"query_{phase}_host_overlap_s", f"query_{phase}_overlap_records"
            result[duration_key] = EXISTING.finite_nonnegative(summary[duration_key])
            result[count_key] = int(summary[count_key])
            if not math.isclose(interval, result[duration_key], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Reported host overlap disagrees with the per-record timeline")
            if count != result[count_key]:
                raise ValueError("Reported overlapping query count disagrees with the timeline")
            if overlap == "off" and interval > 1e-9:
                raise ValueError("OFF mode unexpectedly overlaps query and completed consumer intervals")
        result["host_overlap_observed"] = any(result[f"query_{phase}_host_overlap_s"] > 0
                                               for phase in phases if phase != "dock")
        if args.smoke_only:
            check_full_pool(args, records, result)
        if any(output.glob("*.sdf")):
            raise ValueError("Unexpected SDF output")
    except (OSError, KeyError, ValueError, TypeError) as error:
        result["status"] = "incomplete"
        result["issues"].append(str(error))
        for key in ("successful_records_per_s", "completed_records_per_s", "successful_online_records_per_s"):
            result[key] = None
    return result


def execute(args, repeat, threads, overlap):
    directory = args.output / f"repeat-{repeat}-threads-{threads}-{overlap}"
    directory.mkdir()
    output = directory / "docking"
    environment = run_environment(args, threads)
    argv = command(args, overlap, output)
    EXISTING.write_json(directory / "command.json", {"argv": argv,
        "environment": EXISTING.relevant_environment(environment)})
    start, code = time.perf_counter(), None
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        try:
            code = subprocess.run(argv, cwd=REPOSITORY, env=environment, stdout=stdout, stderr=stderr,
                                  timeout=args.timeout_s, check=False).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stderr.write(str(error) + "\n")
    result = inspect_output(args, output, code, time.perf_counter() - start, threads, overlap)
    result.update(repeat=repeat, directory=directory.name)
    return result


def tables(results, args):
    rows = []
    for result in results:
        baseline = next((r for r in results if r["repeat"] == result["repeat"] and
                         r["threads"] == result["threads"] and r["overlap"] == "off"), None)
        valid = (baseline is not None and result["status"] != "incomplete" and baseline["status"] != "incomplete"
                 and result["comparison"]["equal"] and baseline["comparison"]["equal"]
                 and (result["overlap"] == "off" or result["host_overlap_observed"] or args.smoke_only))
        rows.append({"repeat": result["repeat"], "query_threads": result["threads"], "overlap": result["overlap"],
            "status": result["status"], "correct": result["comparison"]["equal"],
            "processed": result["completed_records"], "successful": result["successful_records"],
            "errors": result["failed_records"], "external_wall_s": result["external_wall_s"],
            "online_wall_s": result["online_wall_s"],
            "successful_ligands_per_s": result["successful_online_records_per_s"],
            "completed_records_per_s": result.get("completed_online_records_per_s"),
            "query_preparation_work_s": result["timings"].get("query_preparation_s", {}).get("sum_s"),
            "gpu_search_wrapper_work_s": result["timings"].get("gpu_search_s", {}).get("sum_s"),
            "consumer_wait_s": result.get("consumer_wait_s"),
            "producer_queue_wait_s": result.get("producer_queue_wait_s"),
            "host_query_search_overlap_s": result.get("query_search_host_overlap_s"),
            "host_overlap_observed": result["host_overlap_observed"],
            "actual_query_threads": json.dumps(result.get("query_thread_use", {}).get("counts", {}), sort_keys=True),
            "online_speedup_vs_same_thread_OFF": baseline["online_wall_s"] / result["online_wall_s"] if valid else None})
        rows[-1]["repetition_scope"] = repetition_scope(args)
        if args.ranking != "fast":
            rows[-1].update(ranking=args.ranking, rescoring_backend=args.backend,
                fast_prefilter=args.fast_prefilter,
                rescoring_wrapper_work_s=result["timings"].get("rescoring_completed_s", {}).get("sum_s"),
                dock_completed_work_s=result["timings"].get("dock_completed_s", {}).get("sum_s"),
                weighted_interaction_work_s=result["timings"].get("weighted_interaction_scoring_s", {}).get("sum_s"),
                vinardo_work_s=result["timings"].get("vinardo_scoring_s", {}).get("sum_s"),
                hybrid_normalization_work_s=result["timings"].get("hybrid_normalization_s", {}).get("sum_s"),
                host_query_rescoring_overlap_s=result.get("query_rescoring_host_overlap_s"),
                host_query_dock_overlap_s=result.get("query_dock_host_overlap_s"))
    save_csv(args.output / "runs.csv", rows)
    summary = []
    for threads in args.threads:
        for overlap in args.overlap:
            group = [r for r in rows if r["query_threads"] == threads and r["overlap"] == overlap]
            # A missing OFF baseline affects speedup, not measured wall/work.
            valid = [r for r in group if r["status"] != "incomplete" and r["correct"]]
            row = {"query_threads": threads, "overlap": overlap, "runs": len(group), "valid_repeats": len(valid)}
            actual = sorted({int(key) for run in valid for key in json.loads(run["actual_query_threads"])})
            row["actual_query_threads_used"] = json.dumps(actual)
            row["all_queries_used_serial_fallback"] = actual == [1] if actual else None
            keys = ("online_wall_s", "external_wall_s", "successful_ligands_per_s",
                        "query_preparation_work_s", "gpu_search_wrapper_work_s", "consumer_wait_s",
                        "producer_queue_wait_s", "host_query_search_overlap_s", "online_speedup_vs_same_thread_OFF")
            if args.ranking != "fast":
                row.update(ranking=args.ranking, rescoring_backend=args.backend, fast_prefilter=args.fast_prefilter)
                keys += ("rescoring_wrapper_work_s", "dock_completed_work_s", "weighted_interaction_work_s",
                         "vinardo_work_s", "hybrid_normalization_work_s",
                         "host_query_rescoring_overlap_s", "host_query_dock_overlap_s")
            for key in keys:
                values = [r[key] for r in valid if r[key] is not None]
                row[key + "_median"] = statistics.median(values) if values else None
            row["repetition_scope"] = repetition_scope(args)
            summary.append(row)
    save_csv(args.output / "summary.csv", summary)


def run_campaign(args):
    for name in ("protein", "pocket", "ligands", "executable", "cmake_cache"):
        if not getattr(args, name).is_file():
            raise ValueError(f"Missing {name}: {getattr(args, name)}")
    if not os.access(args.executable, os.X_OK):
        raise ValueError("Executable is not executable")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    resources, metadata = check_resources(args), build_metadata(args)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata.update(resources=resources, created_utc=datetime.now(timezone.utc).isoformat())
    EXISTING.write_json(args.output / "campaign.json", metadata)
    results, reference = [], None
    runs = ([(0, 1, "legacy")] if args.smoke_only else []) + list(configurations(args))
    for repeat, threads, overlap in runs:
        print(f"repeat={repeat} query_threads={threads} overlap={overlap}", flush=True)
        result = execute(args, repeat, threads, overlap)
        output = args.output / result["directory"] / "docking"
        if reference is None:
            reference = output
        result["comparison"] = EXISTING.compare_tables(reference, output,
            args.absolute_tolerance, args.relative_tolerance)
        EXISTING.write_json(output.parent / "result.json", result)
        results.append(result)
        EXISTING.write_json(args.output / "runs.json", results)
        tables(results, args)
        if result["status"] == "incomplete" or not result["comparison"]["equal"]:
            EXISTING.write_json(args.output / "status.json", {"status": "stopped_on_failure_or_mismatch"})
            return 1
    on_results = [r for r in results if r["overlap"] == "on"]
    demonstrated = bool(on_results) and all(r["host_overlap_observed"] for r in on_results)
    complete = args.smoke_only or not on_results or demonstrated
    EXISTING.write_json(args.output / "status.json", {"status": "complete" if complete else "complete_no_host_overlap",
        "host_overlap_demonstrated": demonstrated, "device_overlap_validation": "requires a separate CUDA/NVTX timeline"})
    return 0 if complete else 1


def main(argv=None):
    try:
        return run_campaign(arguments(argv))
    except (OSError, ValueError, KeyError) as error:
        print(f"Overlap benchmark: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
