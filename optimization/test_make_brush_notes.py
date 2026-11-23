"""Tests for optimization.tools.make_brush_notes (note generator)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from optimization.tools import make_brush_notes as mbn


def make_report(*, clip=False, status="ok", final_ply="out.ply",
                needle=True):
    """Small but complete pipeline report fixture (real schema keys)."""
    clip_stage = ({"name": "mirror_clip", "status": "ok", "time_s": 0.3,
                   "error": None} if clip else
                  {"name": "mirror_clip", "status": "skipped", "time_s": 0.0,
                   "error": None, "reason": "not requested"})
    return {
        "schema_version": 1,
        "status": status,
        "input": {"path": "orig.ply", "size_bytes": 1_000_000,
                  "gaussians": 1000, "sh_degree": 3},
        "stages": [
            {"name": "inspect", "status": "ok", "time_s": 0.1, "error": None},
            {"name": "smart_prune", "status": "ok", "time_s": 0.2,
             "error": None},
            clip_stage,
            {"name": "sh_truncate", "status": "skipped", "time_s": 0.0,
             "error": None, "reason": "not requested (sh_degree=None)"},
            {"name": "normalize", "status": "ok", "time_s": 0.4,
             "error": None},
            {"name": "route_b", "status": "skipped", "time_s": 0.0,
             "error": None, "reason": "not requested"},
            {"name": "report", "status": "ok", "time_s": 0.0, "error": None},
        ],
        "smart_prune": {
            "original_count": 1000,
            "removed_count": 200,
            "pruned_count": 800,
            "removed_by_reason": {"low_importance": 180, "isolated": 15,
                                  "needle": 5 if needle else 0,
                                  "outside_body": 0, "size_outlier": 0},
            "voxels_emptied": 0,
            "thresholds": {"needle_enabled": needle},
        },
        "mirror_clip": ({
            "removed_count": 120,
            "removed_count_share": 0.15,
            "removed_weight_share": 0.445,
            "plane_confidence": 0.414,
            "normal_flipped": True,
            "margin": 1.75,
            "subject_checked": True,
            "max_weight_fraction": 0.6,
        } if clip else None),
        "normalize": {
            "plan": {
                "methods": {"center": "subject density centre",
                            "upright": "estimated up",
                            "scale": "uniform scale p90"},
                "confidence": {"center": 1.0, "upright": 0.42,
                               "scale": 0.9},
                "up_angle_to_brush_y_deg": 3.5,
                "warnings": ["upright confidence low (0.420)"],
            },
            "relative": 6.232769120219085e-08,
            "sidecar_path": "out.normalize.json",
        },
        "counts": {"input": 1000, "pruned": 800,
                   "clipped": 680 if clip else None,
                   "output": 680 if clip else 800},
        "sizes": {"input_bytes": 1_000_000, "final_bytes": 600_000},
        "ratios": {"final_vs_input": 1.667},
        "timings": {"total_s": 2.5},
        "outputs": {"final_ply": final_ply},
        "warnings": ["route_b skipped: not requested"],
    }


def write_json(dirpath, name, data):
    path = Path(dirpath) / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestVerdicts(unittest.TestCase):
    def test_missing_label_defaults_to_not_tested(self):
        self.assertEqual(mbn.format_verdict({}, "N1"), "not tested")

    def test_string_verdict_passes_through(self):
        self.assertEqual(mbn.format_verdict({"N1": "opens ok"}, "N1"),
                         "opens ok")

    def test_dict_verdict_composes_status_gpu_detail(self):
        verdicts = {"original": {"status": "opens ok",
                                 "gpu": "Test GPU (DiscreteGpu)",
                                 "detail": "observed thing"}}
        self.assertEqual(
            mbn.format_verdict(verdicts, "original"),
            "opens ok; GPU: Test GPU (DiscreteGpu); observed thing")

    def test_committed_verdicts_file_is_valid_and_matches_facts(self):
        verdicts = mbn.load_verdicts(mbn.DEFAULT_VERDICTS)
        original = verdicts["original"]
        self.assertIsInstance(original, dict)
        self.assertEqual(original["status"], "opens ok")
        self.assertIn("NVIDIA GeForce RTX 4060 Laptop GPU", original["gpu"])
        for label in ("N1", "N2", "N3", "N4", "smart_default", "centring"):
            self.assertEqual(verdicts[label], "not tested", label)

    def test_load_verdicts_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            mbn.load_verdicts("no/such/verdicts.json")

    def test_load_verdicts_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(tmp, "v.json", {"not_verdicts": {}})
            with self.assertRaises(ValueError):
                mbn.load_verdicts(path)


class TestBuildNote(unittest.TestCase):
    def test_note_carries_report_numbers_and_default_verdict(self):
        report = make_report(clip=True)
        note = mbn.build_note("N4", report, {}, "r.json", "v.json")
        self.assertIn("# N4 - smart_prune + mirror_clip + normalize", note)
        self.assertIn("File: `out.ply` (680 Gaussians, 0.60 MB, "
                      "1.667x vs input)", note)
        self.assertIn("Counts: input 1000 -> pruned 800 -> clipped 680 -> "
                      "output 680", note)
        self.assertIn("Smart prune: removed 200 of 1000 (by reason: "
                      "low_importance 180, isolated 15, needle 5, "
                      "outside_body 0, size_outlier 0); voxels emptied 0; "
                      "needle cleanup on (5 removed).", note)
        self.assertIn("Mirror clip: removed 120 (count share 0.150, "
                      "weight share 0.445); plane confidence 0.414; "
                      "normal_flipped true; margin 1.750; subject checked "
                      "true; weight fraction limit 0.6.", note)
        self.assertIn("sidecar round-trip rel 6.233e-08", note)
        self.assertIn("upright confidence low (0.420)", note)
        self.assertIn("Warnings: route_b skipped: not requested", note)
        self.assertIn("Total time: 2.5 s.", note)
        self.assertIn(".\\optimization\\tools\\open_in_brush.ps1 -Ply "
                      "out.ply", note)
        self.assertIn("Verdict: not tested", note)

    def test_note_omits_clip_section_when_stage_off(self):
        note = mbn.build_note("N1", make_report(clip=False), {}, "r.json",
                              "v.json")
        self.assertNotIn("Mirror clip:", note)
        self.assertIn("clipped n/a (clip off)", note)
        self.assertIn("mirror_clip skipped (not requested) 0.0 s", note)
        self.assertIn("# N1 - smart_prune + normalize", note)
        self.assertIn("needle cleanup on (5 removed)", note)

    def test_note_uses_verdict_from_file_not_a_guess(self):
        verdicts = {"N2": {"status": "opens ok", "gpu": "Some GPU",
                           "detail": "saw the figure centred"}}
        note = mbn.build_note("N2", make_report(), verdicts, "r.json",
                              "v.json")
        self.assertIn("Verdict: opens ok; GPU: Some GPU; saw the figure "
                      "centred", note)

    def test_needle_cleanup_off_line(self):
        note = mbn.build_note("N1", make_report(needle=False), {}, "r.json",
                              "v.json")
        self.assertIn("needle cleanup off", note)


class TestCli(unittest.TestCase):
    def test_cli_writes_note_with_file_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = write_json(tmp, "r.json", make_report(clip=True))
            verdicts = write_json(tmp, "v.json",
                                  {"schema_version": 1,
                                   "verdicts": {"N4": "opens ok; GPU: g"}})
            out = Path(tmp) / "sub" / "N4.md"
            rc = mbn.main(["--note", f"N4={report}={out}",
                           "--verdicts", str(verdicts)])
            self.assertEqual(rc, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("Verdict: opens ok; GPU: g", text)
            self.assertIn("plane confidence 0.414", text)

    def test_cli_bad_note_format_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            verdicts = write_json(tmp, "v.json",
                                  {"schema_version": 1, "verdicts": {}})
            rc = mbn.main(["--note", "only_two=parts",
                           "--verdicts", str(verdicts)])
            self.assertEqual(rc, 2)

    def test_cli_missing_report_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            verdicts = write_json(tmp, "v.json",
                                  {"schema_version": 1, "verdicts": {}})
            rc = mbn.main(["--note", f"N1={tmp}/absent.json={tmp}/n.md",
                           "--verdicts", str(verdicts)])
            self.assertEqual(rc, 2)

    def test_cli_missing_verdicts_file_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = write_json(tmp, "r.json", make_report())
            rc = mbn.main(["--note", f"N1={report}={tmp}/n.md",
                           "--verdicts", f"{tmp}/absent.json"])
            self.assertEqual(rc, 2)

    def test_cli_error_status_report_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = write_json(tmp, "r.json",
                                make_report(status="error"))
            verdicts = write_json(tmp, "v.json",
                                  {"schema_version": 1, "verdicts": {}})
            rc = mbn.main(["--note", f"N1={report}={tmp}/n.md",
                           "--verdicts", str(verdicts)])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
