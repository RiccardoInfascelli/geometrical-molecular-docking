"""Check saved-pool Hybrid orchestration using synthetic files only."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rank_hybrid


def write_csv(path, fields, data):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


class SavedHybridPoolTest(unittest.TestCase):
    def setUp(self):
        runs = Path(__file__).resolve().parent / "runs"
        runs.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-hybrid-", dir=runs)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.preparation = self.source / "preparation"
        self.docking = self.source / "docking"
        self.preparation.mkdir(parents=True)
        self.docking.mkdir()
        self.source_protocol = {"docking_exit_status": 0, "complex": "/synthetic/1a1e",
                                "status": "ok"}
        self.metadata = [
            {"record_index": "1", "molecule_id": "baseline", "source": "crystal_original"},
            {"record_index": "2", "molecule_id": "conformer_001", "source": "etkdgv3"},
            {"record_index": "3", "molecule_id": "conformer_002", "source": "etkdgv3"},
        ]
        self.records = [
            {"record_index": "1", "molecule_id": "baseline", "status": "ok", "survivors": "2"},
            {"record_index": "2", "molecule_id": "conformer_001", "status": "ok", "survivors": "12"},
            {"record_index": "3", "molecule_id": "conformer_002", "status": "ok", "survivors": "3"},
        ]
        self.protocol = {
            "pose_pool": "all operational clash survivors; no prefilter or deduplication",
            "index_policy": "shared_index_only; cutoff=24", "record_count": "3"}
        self.poses = []
        for record in self.records:
            for pose_id in range(int(record["survivors"])):
                row = {field: "0" for field in rank_hybrid.POSE_FIELDS}
                row.update(record_index=record["record_index"], molecule_id=record["molecule_id"],
                           pose_id=str(pose_id), fast_score=str(100 - pose_id),
                           rmsd_heavy_A="3.25", r00="1", r11="1", r22="1")
                self.poses.append(row)
        self.poses[13]["fast_score"] = "-999"  # A legitimate pose outside Fast Top-10.
        self.write_source()
        for name in ("conformers.mol2", "reference.mol2"):
            (self.preparation / name).write_text("Synthetic placeholder; never parsed by these tests.\n")
        self.driver = self.root / "conformer_hybrid"
        self.driver.write_text("Synthetic driver; subprocess is mocked.\n")
        self.tsv_counter = 0

    def write_source(self):
        (self.source / "protocol.json").write_text(json.dumps(self.source_protocol))
        write_csv(self.preparation / "conformers.csv", list(self.metadata[0]), self.metadata)
        write_csv(self.docking / "records.csv", list(self.records[0]), self.records)
        write_csv(self.docking / "protocol.csv", ["key", "value"],
                  [{"key": key, "value": value} for key, value in self.protocol.items()])
        write_csv(self.docking / "poses.csv", rank_hybrid.POSE_FIELDS + ["molecule_id"], self.poses)

    def snapshot(self):
        return {str(path.relative_to(self.source)): path.read_bytes()
                for path in self.source.rglob("*") if path.is_file()}

    def export(self):
        self.tsv_counter += 1
        output = self.root / f"input-{self.tsv_counter}.tsv"
        return rank_hybrid.write_pose_input(self.source, output), output

    def main_arguments(self, output):
        return ["--run", str(self.source), "--driver", str(self.driver), "--output", str(output)]

    def test_all_survivors_retained_and_baseline_counted_separately(self):
        before = self.snapshot()
        metadata, exported = self.export()
        self.assertEqual(metadata, {"source_complex": "/synthetic/1a1e", "record_count": 3,
                                   "conformer_count": 2, "baseline_poses": 2, "ensemble_poses": 15})
        with exported.open(newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            self.assertEqual(reader.fieldnames, rank_hybrid.POSE_FIELDS)
            actual = list(reader)
        self.assertEqual(len(actual), 17)
        self.assertEqual(actual, [{field: row[field] for field in rank_hybrid.POSE_FIELDS}
                                  for row in self.poses])
        self.assertEqual(sum(row["record_index"] == "1" for row in actual), 2)
        self.assertEqual(actual[13]["fast_score"], "-999")
        self.assertNotIn(b"\r", exported.read_bytes())
        self.assertEqual(self.snapshot(), before)

    def test_completed_search_can_survive_an_unrelated_analysis_failure(self):
        self.source_protocol["status"] = "failed"
        self.write_source()
        self.assertEqual(self.export()[0]["ensemble_poses"], 15)

    def test_zero_survivor_record_is_valid(self):
        self.records[2].update(status="no_poses", survivors="0")
        self.poses = [row for row in self.poses if row["record_index"] != "3"]
        self.write_source()
        metadata, _ = self.export()
        self.assertEqual(metadata["conformer_count"], 2)
        self.assertEqual(metadata["ensemble_poses"], 12)

    def test_duplicate_and_out_of_range_pose_ids_are_rejected(self):
        original = [dict(row) for row in self.poses]
        for invalid in ("0", "12"):
            with self.subTest(pose_id=invalid):
                self.poses = [dict(row) for row in original]
                self.poses[3]["pose_id"] = invalid
                self.write_source()
                with self.assertRaisesRegex(ValueError, "Duplicate or out-of-range"):
                    self.export()

    def test_truncated_pose_export_is_rejected(self):
        self.poses.pop()
        self.write_source()
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.export()

    def test_incomplete_records_are_rejected(self):
        self.records.pop()
        self.write_source()
        with self.assertRaisesRegex(ValueError, "records are incomplete"):
            self.export()

    def test_failed_search_and_invalid_pool_protocol_are_rejected(self):
        for status in (1, "0", True):
            with self.subTest(exit_status=status):
                self.source_protocol["docking_exit_status"] = status
                self.write_source()
                with self.assertRaisesRegex(ValueError, "exit successfully"):
                    self.export()
        self.source_protocol["docking_exit_status"] = 0
        original = dict(self.protocol)
        for key, value in (("pose_pool", "Fast top-10"), ("index_policy", "fallback"),
                           ("record_count", "2")):
            with self.subTest(key=key):
                self.protocol = dict(original)
                self.protocol[key] = value
                self.write_source()
                with self.assertRaises(ValueError):
                    self.export()

    def test_baseline_and_generated_metadata_must_remain_distinct(self):
        self.metadata[1]["source"] = "crystal_original"
        self.write_source()
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.export()

    def test_nonfinite_and_invalid_numeric_fields_are_rejected(self):
        original = dict(self.poses[-1])
        for field, value in (("fast_score", "nan"), ("r00", "inf"), ("tx", "-inf"),
                             ("rmsd_heavy_A", "-0.1"), ("clash_fraction", "1.1"),
                             ("occupied_atoms", "-1"), ("pose_id", "1junk")):
            with self.subTest(field=field, value=value):
                self.poses[-1] = dict(original)
                self.poses[-1][field] = value
                self.write_source()
                with self.assertRaises(ValueError):
                    self.export()

    def test_wrong_conformer_identity_and_duplicate_records_are_rejected(self):
        self.poses[-1]["molecule_id"] = "baseline"
        self.write_source()
        with self.assertRaisesRegex(ValueError, "declared conformer"):
            self.export()
        self.poses[-1]["molecule_id"] = "conformer_002"
        self.records.append(dict(self.records[-1]))
        self.write_source()
        with self.assertRaisesRegex(ValueError, "duplicate docking record"):
            self.export()

    def test_existing_export_is_never_overwritten(self):
        output = self.root / "existing.tsv"
        output.write_bytes(b"keep this file intact\n")
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            rank_hybrid.write_pose_input(self.source, output)
        self.assertEqual(output.read_bytes(), b"keep this file intact\n")
        self.assertEqual(self.snapshot(), before)

    def test_existing_analysis_directory_is_protected(self):
        output = self.root / "existing-analysis"
        output.mkdir()
        (output / "protocol.json").write_text("preserve previous protocol\n")
        with patch.object(rank_hybrid.subprocess, "run") as run, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(rank_hybrid.main(self.main_arguments(output)), 1)
        run.assert_not_called()
        self.assertEqual((output / "protocol.json").read_text(), "preserve previous protocol\n")

    def test_summary_requires_top_ten_and_sufficient_local_candidates(self):
        for local_k, top_x in ((10, 5), (5, 10)):
            with self.subTest(local_k=local_k, top_x=top_x):
                output = self.root / f"invalid-topk-{local_k}-{top_x}"
                arguments = self.main_arguments(output) + ["--local-k", str(local_k), "--top-x", str(top_x)]
                with patch.object(rank_hybrid.subprocess, "run") as run, \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(rank_hybrid.main(arguments), 1)
                run.assert_not_called()
                self.assertFalse(output.exists())

    def test_failed_rescoring_is_recorded_without_rerunning_geometry(self):
        before = self.snapshot()
        output = self.root / "failed-analysis"
        with patch.object(rank_hybrid.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(rank_hybrid.main(self.main_arguments(output)), 1)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [str(self.driver.resolve()), "/synthetic/1a1e",
                                      str(self.preparation.resolve() / "conformers.mol2"),
                                      str(self.preparation.resolve() / "reference.mol2")])
        self.assertEqual(command[4:6], [str((output / "pose_input.tsv").resolve()),
                                       str((output / "ranking").resolve())])
        self.assertEqual(command[6:], ["--interaction-region", "pocket", "--local-k", "10", "--top-x", "10"])
        self.assertNotIn("conformer_docking", " ".join(command))
        self.assertNotIn("prepare_conformers", " ".join(command))
        protocol = json.loads((output / "protocol.json").read_text())
        self.assertEqual(protocol["status"], "failed")
        self.assertEqual(protocol["rescoring_exit_status"], 7)
        self.assertFalse(protocol["geometric_search_rerun"])
        self.assertFalse(protocol["source_outputs_modified"])
        self.assertEqual(protocol["baseline_poses"], 2)
        self.assertEqual(protocol["ensemble_poses"], 15)
        self.assertEqual(protocol["hybrid_lambda"], 0.3)
        self.assertEqual(protocol["vinardo_repulsion_alpha"], 0.5)
        self.assertEqual(self.snapshot(), before)

    def test_successful_rescoring_writes_a_separate_protocol_and_uses_overrides(self):
        before = self.snapshot()
        output = self.root / "successful-analysis"

        def finish(command, **kwargs):
            ranking = Path(command[5])
            ranking.mkdir()
            (ranking / "topk_summary.csv").write_text("group,k,best_rmsd_heavy_A\nensemble,1,1.2\n")
            return SimpleNamespace(returncode=0)

        arguments = self.main_arguments(output) + ["--complex", "/synthetic/alternate/1a1e",
                                                   "--interaction-region", "protein"]
        with patch.object(rank_hybrid.subprocess, "run", side_effect=finish) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rank_hybrid.main(arguments), 0)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[1], "/synthetic/alternate/1a1e")
        self.assertEqual(command[6:8], ["--interaction-region", "protein"])
        protocol = json.loads((output / "protocol.json").read_text())
        self.assertEqual(protocol["status"], "ok")
        self.assertEqual(protocol["rescoring_exit_status"], 0)
        self.assertEqual(self.snapshot(), before)

    def test_success_without_summary_is_a_failure(self):
        output = self.root / "missing-summary"
        with patch.object(rank_hybrid.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(rank_hybrid.main(self.main_arguments(output)), 1)
        protocol = json.loads((output / "protocol.json").read_text())
        self.assertEqual(protocol["status"], "failed")
        self.assertIn("without topk_summary.csv", protocol["error"])


if __name__ == "__main__":
    unittest.main()
