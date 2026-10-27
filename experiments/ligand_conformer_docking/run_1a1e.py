#!/usr/bin/env python3
"""Run the conformer experiment once each for 1, 5, 10 and 20 conformers."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
COUNTS = (1, 5, 10, 20)


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def positive_float(value):
    number = nonnegative_float(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def prefilter(value):
    return value if value == "none" else str(positive_integer(value))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("complex", "executable", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ligand", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--min-rmsd", type=nonnegative_float, default=0.5)
    parser.add_argument("--mmff", action="store_true")
    parser.add_argument("--max-iterations", type=positive_integer, default=500)
    parser.add_argument("--fast-prefilter", type=prefilter, default="none")
    parser.add_argument("--crop-margin", choices=("off", "8"), default="off")
    parser.add_argument("--cutoff", type=positive_float, default=24.0)
    parser.add_argument("--write-sdf", action="store_true")
    args = parser.parse_args(argv)
    for name in ("complex", "executable", "output", "ligand", "reference"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


def selected_environment(environment):
    selected = {"SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST"}
    return {key: value for key, value in sorted(environment.items())
            if key.startswith(("DOCKING_", "OMP_", "CUDA_")) or key in selected}


def git_command(*flags):
    try:
        result = subprocess.run(["git", *flags], cwd=REPOSITORY, capture_output=True,
                                text=True, timeout=20, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def provenance():
    return {"head": git_command("rev-parse", "HEAD"),
            "branch": git_command("branch", "--show-current"),
            "status": git_command("status", "--short"),
            "diff": git_command("diff", "--binary", "HEAD")}


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def code_provenance(executable, directory=None):
    directory = directory if directory is not None else Path(__file__).resolve().parent
    sources = {path.name: file_identity(path) for path in sorted(directory.iterdir())
               if path.is_file() and (path.suffix in (".cpp", ".hpp", ".py")
                                      or path.name == "CMakeLists.txt")}
    return {"executable": file_identity(executable), "sources": sources}


def command(args, count, output):
    result = [str(args.executable), str(args.complex), str(output),
              "--num-conformers", str(count), "--seed", str(args.seed),
              "--min-rmsd", str(args.min_rmsd), "--max-iterations", str(args.max_iterations),
              "--fast-prefilter", args.fast_prefilter, "--crop-margin", args.crop_margin,
              "--cutoff", str(args.cutoff)]
    for name in ("ligand", "reference"):
        if getattr(args, name) is not None:
            result.extend(["--" + name, str(getattr(args, name))])
    if args.mmff:
        result.append("--mmff")
    if args.write_sdf:
        result.append("--write-sdf")
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_progress(args, protocol):
    write_json(args.output / "protocol.json", protocol)
    fields = ("num_conformers", "status", "returncode", "external_wall_s", "output", "error")
    with (args.output / "runs.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(protocol["runs"])


def run(args):
    if not args.complex.is_dir():
        raise ValueError(f"Missing complex directory: {args.complex}")
    if not args.executable.is_file() or not os.access(args.executable, os.X_OK):
        raise ValueError(f"Missing executable or execute permission: {args.executable}")
    for name in ("ligand", "reference"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            raise ValueError(f"Missing {name}: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    protocol = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "counts": list(COUNTS), "repeats": 1,
                "options": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                "selected_environment": selected_environment(environment),
                "git": provenance(), "code": code_provenance(args.executable), "runs": []}
    save_progress(args, protocol)
    for count in COUNTS:
        directory = args.output / f"conformers-{count}"
        directory.mkdir(exist_ok=False)
        output = directory / "docking"  # The executable requires a new directory.
        argv = command(args, count, output)
        write_json(directory / "command.json", {"argv": argv, "cwd": str(REPOSITORY),
                   "selected_environment": selected_environment(environment)})
        entry = {"num_conformers": count, "status": "failed", "returncode": None,
                 "output": str(output.relative_to(args.output)), "command": argv, "error": ""}
        print(f"Requested conformers: {count}", flush=True)
        start = time.perf_counter()
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            try:
                result = subprocess.run(argv, cwd=REPOSITORY, env=environment,
                                        stdout=stdout, stderr=stderr, check=False)
                entry["returncode"] = result.returncode
            except OSError as error:
                entry["error"] = str(error)
                stderr.write(str(error) + "\n")
        entry["external_wall_s"] = time.perf_counter() - start
        if entry["returncode"] == 0 and (output / "summary.csv").is_file():
            entry["status"] = "complete"
        elif not entry["error"]:
            entry["error"] = (f"Executable exited {entry['returncode']}" if entry["returncode"] != 0
                              else "Executable returned zero without summary.csv")
        protocol["runs"].append(entry)
        if entry["status"] != "complete":
            protocol["status"] = "failed"
            save_progress(args, protocol)
            print(f"Stopped: {entry['error']}. Partial outputs and logs: {directory}", file=sys.stderr)
            return 1
        save_progress(args, protocol)
    protocol["status"] = "complete"
    protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save_progress(args, protocol)
    print(f"Four configurations completed: {args.output}", flush=True)
    return 0


def main(argv=None):
    try:
        return run(arguments(argv))
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
