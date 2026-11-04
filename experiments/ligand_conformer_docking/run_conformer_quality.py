#!/usr/bin/env python3
"""Run one CPU conformer-quality sequence per explicitly listed complex."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


REQUIRED_OUTPUTS = ("conformers.csv", "budgets.csv", "summary.csv", "atom_mapping.csv", "protocol.json")
ENGINE_STATUSES = {"ok", "incomplete", "generation_failed", "stereo_ambiguous", "error"}
BUDGETS = (1, 5, 10, 20, 50, 100, 200)


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def conformer_count(value):
    number = positive_integer(value)
    if number > 200:
        raise argparse.ArgumentTypeError("must be at most 200")
    return number


def seed_value(value):
    number = int(value)
    if number < 0 or number > 2147483647:
        raise argparse.ArgumentTypeError("must be between 0 and 2147483647")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-list", type=Path, required=True,
                        help="plaintext complex IDs or directories, one per line")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="new, nonexistent batch directory")
    destination.add_argument("--resume", type=Path, help="existing compatible batch directory")
    parser.add_argument("--limit", type=positive_integer)
    parser.add_argument("--max-conformers", type=conformer_count, default=200)
    parser.add_argument("--seed", type=seed_value, default=20261027,
                        help="same explicit seed for every complex")
    parser.add_argument("--allow-undefined-stereo", action="store_true")
    parser.add_argument("--no-coordinates", action="store_true",
                        help="omit per-conformer coordinate CSV; all RMSD metrics remain available")
    args = parser.parse_args(argv)
    for name in ("complex_list", "dataset_root", "executable", "output", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.resume is not None:
        args.output = args.resume
    return args


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, writer):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, prefix="." + path.name + ".",
                                         delete=False) as target:
            temporary = Path(target.name)
            writer(target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_write(path, lambda target: json.dump(value, target, indent=2, allow_nan=False))


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def executable_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def input_stat(path):
    try:
        value = path.stat()
        return {"exists": True, "size": value.st_size, "mtime_ns": value.st_mtime_ns}
    except FileNotFoundError:
        return {"exists": False}


def read_cases(path, dataset_root):
    cases, ids = [], set()
    with path.open(encoding="utf-8") as source:
        for line_number, text in enumerate(source, start=1):
            text = text.strip()
            if not text or text.startswith("#"):
                continue
            directory = Path(text).expanduser()
            if not directory.is_absolute():
                directory = dataset_root / directory
            case_id = directory.name
            if not case_id or case_id in ids:
                raise ValueError(f"Empty or duplicate case ID at list line {line_number}: {case_id!r}")
            ids.add(case_id)
            directory = directory.resolve()
            cases.append({"case_id": case_id, "directory": str(directory),
                          "ligand": str(directory / (case_id + "_ligand.mol2"))})
    if not cases:
        raise ValueError("Complex list contains no cases")
    return cases


def expected_protocol(args, cases):
    selected = cases[:args.limit] if args.limit is not None else cases
    return {
        "version": 1, "complex_list": str(args.complex_list),
        "dataset_root": str(args.dataset_root), "executable": executable_identity(args.executable),
        "seed": args.seed, "seed_policy": "same explicit seed for every complex",
        "max_conformers": args.max_conformers, "allow_undefined_stereo": args.allow_undefined_stereo,
        "coordinates_export": not args.no_coordinates,
        "limit": args.limit, "cases": cases,
        "selected_inputs": [{**case, "stat": input_stat(Path(case["ligand"]))} for case in selected],
    }


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields, rows = reader.fieldnames, list(reader)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Missing or duplicate CSV header: {path.name}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path.name}")
    return fields, rows


def inspect_outputs(directory):
    missing = [name for name in REQUIRED_OUTPUTS if not (directory / name).is_file()]
    if missing:
        raise ValueError("Missing output files: " + ", ".join(missing))
    summary_fields, summaries = read_csv(directory / "summary.csv")
    budget_fields, budgets = read_csv(directory / "budgets.csv")
    if len(summaries) != 1 or summaries[0].get("status") not in ENGINE_STATUSES:
        raise ValueError("summary.csv must contain one recognized terminal status")
    if "budget" not in budget_fields or [row["budget"] for row in budgets] != [str(value) for value in BUDGETS]:
        raise ValueError("budgets.csv must contain the seven ordered budgets 1/5/10/20/50/100/200")
    protocol = read_json(directory / "protocol.json")
    if not isinstance(protocol, dict) or protocol.get("status") != summaries[0]["status"]:
        raise ValueError("Engine protocol.json status disagrees with summary.csv")
    return {"summary_fields": summary_fields, "summary": summaries[0],
            "budget_fields": budget_fields, "budgets": budgets}


def case_directory(root, index, case_id):
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id).strip(".") or "case"
    return root / "cases" / f"{index:06d}-{safe_id}"


def attempts(directory):
    if not directory.exists():
        return []
    return sorted((path for path in directory.iterdir()
                   if path.is_dir() and re.fullmatch(r"attempt-\d+", path.name)),
                  key=lambda path: int(path.name.split("-")[1]))


def completed_attempt(directory):
    for attempt in reversed(attempts(directory)):
        try:
            outputs = inspect_outputs(attempt / "result")
            metadata = read_json(attempt / "status.json") if (attempt / "status.json").is_file() else {}
            if not isinstance(metadata, dict):
                metadata = {}
            # A complete engine summary can survive an interruption before the
            # wrapper's final status write. Never invent a recovered exit code.
            if metadata.get("phase") != "complete":
                metadata.update(phase="complete", recovered_from_outputs=True,
                                returncode=metadata.get("returncode"))
                write_json(attempt / "status.json", metadata)
            return attempt, metadata, outputs
        except (OSError, ValueError, TypeError):
            continue
    return None


def command(args, case, output):
    argv = [str(args.executable), "--ligand", case["ligand"], "--output", str(output),
            "--max-conformers", str(args.max_conformers), "--seed", str(args.seed)]
    if args.allow_undefined_stereo:
        argv.append("--allow-undefined-stereo")
    if args.no_coordinates:
        argv.append("--no-coordinates")
    return argv


def execute_case(args, case, directory):
    directory.mkdir(parents=True, exist_ok=True)
    previous = attempts(directory)
    number = int(previous[-1].name.split("-")[1]) + 1 if previous else 1
    attempt = directory / f"attempt-{number:03d}"
    attempt.mkdir(exist_ok=False)
    argv = command(args, case, attempt / "result")
    metadata = {"case_id": case["case_id"], "phase": "running", "started_utc": utc_now(),
                "command": argv, "returncode": None, "error": ""}
    write_json(attempt / "status.json", metadata)
    started = time.perf_counter()
    interrupted = False
    with (attempt / "stdout.log").open("w") as stdout, (attempt / "stderr.log").open("w") as stderr:
        try:
            result = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
            metadata["returncode"] = result.returncode
        except OSError as error:
            metadata["error"] = str(error)
            stderr.write(str(error) + "\n")
        except KeyboardInterrupt:
            interrupted = True
            metadata["error"] = "Interrupted by user"
    metadata.update(external_wall_s=time.perf_counter() - started, finished_utc=utc_now())
    outputs = None
    try:
        if not interrupted:
            outputs = inspect_outputs(attempt / "result")
    except (OSError, ValueError, TypeError) as error:
        metadata["error"] = "; ".join(filter(None, (metadata["error"], str(error))))
    metadata["phase"] = "interrupted" if interrupted else "complete" if outputs is not None else "failed"
    if outputs is not None:
        metadata["engine_status"] = outputs["summary"]["status"]
    write_json(attempt / "status.json", metadata)
    return attempt, metadata, outputs


def write_table(path, fields, rows):
    def write(target):
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(path, write)


def aggregate(root, entries):
    summaries, budgets, failures = [], [], []
    summary_fields, budget_fields = ["case_id"], ["case_id"]
    for entry in entries:
        case_id, metadata, outputs = entry["case_id"], entry["metadata"], entry["outputs"]
        if outputs is not None:
            for field in outputs["summary_fields"]:
                if field not in summary_fields:
                    summary_fields.append(field)
            for field in outputs["budget_fields"]:
                if field not in budget_fields:
                    budget_fields.append(field)
            summaries.append({"case_id": case_id, **outputs["summary"]})
            budgets.extend({"case_id": case_id, **row} for row in outputs["budgets"])
        status = outputs["summary"]["status"] if outputs is not None else metadata["phase"]
        if status != "ok" or metadata.get("returncode") not in (None, 0):
            failures.append({"case_id": case_id, "attempt": entry["attempt"],
                             "returncode": metadata.get("returncode"), "status": status,
                             "error": metadata.get("error") or (outputs["summary"].get("error", "")
                                       if outputs is not None else "")})
    write_table(root / "summary.csv", summary_fields, summaries)
    write_table(root / "budgets.csv", budget_fields, budgets)
    write_table(root / "failures.csv", ["case_id", "attempt", "returncode", "status", "error"], failures)


def run(args):
    if not args.executable.is_file() or not os.access(args.executable, os.X_OK):
        raise ValueError(f"Missing executable or execute permission: {args.executable}")
    cases = read_cases(args.complex_list, args.dataset_root)
    protocol = expected_protocol(args, cases)
    if args.resume is not None:
        manifest = read_json(args.output / "manifest.json")
        saved_protocol = manifest.get("protocol") if isinstance(manifest, dict) else None
        if isinstance(saved_protocol, dict):
            saved_protocol = {**saved_protocol}
            # Older runs always exported coordinates; do not rewrite their manifest.
            saved_protocol.setdefault("coordinates_export", True)
        if saved_protocol != protocol:
            raise ValueError("Resume protocol differs (inputs/order/stat, executable SHA-256, seed or options)")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "manifest.json", {"created_utc": utc_now(), "protocol": protocol})
    selected = cases[:args.limit] if args.limit is not None else cases
    entries = []
    for index, case in enumerate(selected, start=1):
        write_json(args.output / "status.json", {"phase": "running", "finished_cases": len(entries),
                   "requested_cases": len(selected), "updated_utc": utc_now()})
        directory = case_directory(args.output, index, case["case_id"])
        recovered = completed_attempt(directory) if args.resume is not None else None
        print(f"{case['case_id']}: {'reuse terminal result' if recovered else 'run CPU sequence'}", flush=True)
        attempt, metadata, outputs = recovered or execute_case(args, case, directory)
        entries.append({"case_id": case["case_id"], "attempt": str(attempt.relative_to(args.output)),
                        "metadata": metadata, "outputs": outputs})
        aggregate(args.output, entries)
        if metadata["phase"] == "interrupted":
            write_json(args.output / "status.json", {"phase": "interrupted", "finished_cases": len(entries) - 1,
                       "requested_cases": len(selected), "updated_utc": utc_now()})
            return 130
    errors = sum(entry["outputs"] is None or entry["metadata"].get("returncode") not in (None, 0)
                 or entry["outputs"]["summary"]["status"] == "error" for entry in entries)
    write_json(args.output / "status.json", {"phase": "complete", "finished_cases": len(entries),
               "requested_cases": len(selected), "process_or_output_errors": errors, "updated_utc": utc_now()})
    return 1 if errors else 0


def main(argv=None):
    try:
        return run(arguments(argv))
    except (OSError, ValueError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
