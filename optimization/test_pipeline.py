"""Unit and integration tests for optimization.pipeline.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest optimization.test_pipeline -v
"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from optimization.pipeline import run_pipeline
from optimization.quantize import core as route_b_core
from optimization.quantize import sh_truncate

REPO_ROOT = Path(__file__).resolve().parents[1]
BRUSH_COMMENTS = ["Exported from Brush", "Vertical axis: y"]
REST_COUNTS = {0: 0, 1: 9, 2: 24, 3: 45}


def make_vertex_array(n=400, seed=0, sh_degree=3):
    """Small synthetic Brush-style vertex array.

    f_rest_* carries a deliberate channel-major block structure with
    decaying energy per coefficient so sh_truncate's layout evidence
    detects channel_major. Positions are a dense box in [0, 10]^3 and
    opacity/scale are low, so the default importance threshold flags most
    Gaussians and the voxel budget actually spends removals.
    """
    rng = np.random.default_rng(seed)
    k = REST_COUNTS[sh_degree]
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
    # Brush lists f_rest_* in text-sorted order in the header
    fields += [(name, "<f4") for name in sorted(f"f_rest_{i}" for i in range(k))]
    fields += [("opacity", "<f4"),
               ("rot_0", "<f4"), ("rot_1", "<f4"),
               ("rot_2", "<f4"), ("rot_3", "<f4"),
               ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4")]
    dt = np.dtype(fields)
    v = np.zeros(n, dtype=dt)
    v["x"] = rng.uniform(0.0, 10.0, n).astype("<f4")
    v["y"] = rng.uniform(0.0, 10.0, n).astype("<f4")
    v["z"] = rng.uniform(0.0, 10.0, n).astype("<f4")
    v["opacity"] = rng.normal(-6.0, 1.0, n).astype("<f4")
    for i in range(3):
        v[f"f_dc_{i}"] = rng.normal(0.0, 0.3, n).astype("<f4")
        v[f"scale_{i}"] = rng.normal(-2.0, 0.3, n).astype("<f4")
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(4):
        v[f"rot_{i}"] = q[:, i].astype("<f4")
    if k:
        cp = k // 3
        base = np.exp(-np.arange(cp) / 2.0)
        for c in range(3):
            channel_scale = 1.0 + 0.05 * c
            for i in range(cp):
                sigma = float(np.sqrt(base[i] * channel_scale))
                v[f"f_rest_{c * cp + i}"] = rng.normal(
                    0.0, sigma, n).astype("<f4")
    return v


def write_ply(path, v, sh_degree=None):
    comments = list(BRUSH_COMMENTS)
    if sh_degree is not None:
        comments.append(f"SH degree: {sh_degree}")
    ply = PlyData([PlyElement.describe(v, "vertex")],
                  text=False, byte_order="<", comments=comments)
    ply.write(str(path))
    return path


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestPipelineRun(PipelineTestCase):
    def setUp(self):
        super().setUp()
        self.v = make_vertex_array(400, seed=0, sh_degree=3)
        self.src = write_ply(self.dir / "scene.ply", self.v, sh_degree=3)
        self.original_bytes = self.src.read_bytes()

    def test_full_run_ok(self):
        res = run_pipeline(self.src, self.dir, sh_degree=1, route_b=True,
                           quiet=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        self.assertEqual(res["exit_code"], 0)
        self.assertIsNone(res["error"])

        stages = {s["name"]: s["status"] for s in res["stages"]}
        self.assertEqual(list(stages),
                         ["inspect", "smart_prune", "mirror_clip",
                          "sh_truncate", "normalize", "route_b", "report"])
        # every requested stage ran; the opt-in stages are skipped by default
        for name, status in stages.items():
            if name in ("mirror_clip", "normalize"):
                self.assertEqual(status, "skipped", stages)
            else:
                self.assertEqual(status, "ok", stages)
        for s in res["stages"]:
            self.assertGreaterEqual(s["time_s"], 0.0)

        # the input is never modified
        self.assertEqual(self.src.read_bytes(), self.original_bytes)

        pruned = self.dir / "scene_smart.ply"
        shd = self.dir / "scene_smart_shd1.ply"
        npz = self.dir / "scene_smart_shd1.npz"
        qback = self.dir / "scene_smart_shd1_qback.ply"
        report = self.dir / "scene_pipeline.json"
        csv_log = self.dir / "pipeline_runs.csv"
        for p in (pruned, shd, npz, qback, report, csv_log):
            self.assertTrue(p.is_file(), f"missing output {p.name}")

        # report file == returned dict
        self.assertEqual(json.loads(report.read_text(encoding="utf-8")), res)

        self.assertEqual(res["counts"]["input"], 400)
        self.assertGreater(res["counts"]["pruned"], 0)
        self.assertLess(res["counts"]["pruned"], 400)
        self.assertEqual(res["counts"]["output"], res["counts"]["pruned"])
        self.assertEqual(res["input"]["sh_degree"], 3)
        self.assertEqual(res["sh_truncate"]["degree_out"], 1)

        self.assertGreater(res["sizes"]["input_bytes"],
                           res["sizes"]["final_bytes"])
        self.assertGreater(res["ratios"]["final_vs_input"], 1.0)
        self.assertGreater(res["ratios"]["npz_vs_input"], 1.0)

        # spatial safety: no voxel emptied
        self.assertEqual(res["coverage"]["voxels_emptied"], 0)
        self.assertEqual(res["smart_prune"]["voxels_emptied"], 0)

        # Route B round trip
        self.assertTrue(res["route_b"]["encode_within_step_bound"])
        self.assertGreaterEqual(
            res["route_b"]["dequantize_check"]["max_abs_err_vs_source"], 0.0)
        self.assertEqual(res["route_b"]["settings_bits"]["rest"], 8)

        # q_back is a readable float32 PLY with the same count
        back = PlyData.read(str(qback))["vertex"].data
        self.assertEqual(len(back), res["counts"]["output"])
        self.assertEqual(list(back.dtype.names),
                         list(PlyData.read(str(shd))["vertex"].data.dtype.names))

        # one CSV row
        with open(csv_log, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["sh_degree"], "1")
        self.assertEqual(rows[0]["route_b"], "yes")
        self.assertEqual(rows[0]["normalize"], "no")
        self.assertEqual(rows[0]["mirror_clip"], "no")

    def test_schema_is_stable(self):
        res = run_pipeline(self.src, self.dir, quiet=True)
        expected = {
            "schema_version", "generated_at", "status", "error",
            "error_kind", "error_stage", "exit_code", "input", "project",
            "output_dir", "stages", "inspect", "smart_prune", "mirror_clip",
            "coverage", "sh_truncate", "normalize", "route_b", "counts",
            "sizes", "ratios", "timings", "outputs", "warnings",
        }
        self.assertEqual(set(res), expected)
        self.assertEqual(res["schema_version"], 1)
        self.assertIn("total_s", res["timings"])
        for stage in ("inspect", "smart_prune", "mirror_clip", "sh_truncate",
                      "normalize", "route_b", "report"):
            self.assertIn(stage, res["timings"])
        # no numpy types leak through
        json.dumps(res)

    def test_refuses_to_overwrite(self):
        first = run_pipeline(self.src, self.dir, quiet=True)
        self.assertEqual(first["status"], "ok")
        report = self.dir / "scene_pipeline.json"
        before = report.read_bytes()

        second = run_pipeline(self.src, self.dir, quiet=True)
        self.assertEqual(second["status"], "error")
        self.assertEqual(second["error_kind"], "exists")
        self.assertEqual(second["exit_code"], 2)
        self.assertIn("refusing to overwrite", second["error"])
        # nothing was touched
        self.assertEqual(report.read_bytes(), before)
        self.assertEqual(self.src.read_bytes(), self.original_bytes)

    def test_missing_input(self):
        res = run_pipeline(self.dir / "nope.ply", self.dir, quiet=True)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error_kind"], "bad_input")
        self.assertEqual(res["exit_code"], 2)
        self.assertIn("input not found", res["error"])
        self.assertFalse((self.dir / "nope_pipeline.json").exists())
        self.assertFalse((self.dir / "pipeline_runs.csv").exists())

    def test_bad_option_value(self):
        res = run_pipeline(self.src, self.dir, max_voxel_fraction=0.0,
                           quiet=True)
        self.assertEqual(res["error_kind"], "bad_input")
        self.assertEqual(res["exit_code"], 2)
        self.assertIn("max_voxel_fraction", res["error"])
        self.assertFalse((self.dir / "scene_pipeline.json").exists())

    def test_unknown_option_rejected(self):
        res = run_pipeline(self.src, self.dir, voxel_sizee=3.0, quiet=True)
        self.assertEqual(res["error_kind"], "bad_input")
        self.assertIn("unknown option", res["error"])

    def test_bad_ply_writes_error_report(self):
        bad = self.dir / "bad.ply"
        bad.write_text("this is not a PLY file\n", encoding="utf-8")
        res = run_pipeline(bad, self.dir, quiet=True)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error_kind"], "stage")
        self.assertEqual(res["exit_code"], 3)
        self.assertEqual(res["error_stage"], "inspect")
        report = self.dir / "bad_pipeline.json"
        self.assertTrue(report.is_file())
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "error")
        self.assertIn("could not inspect PLY", payload["error"])

    def test_missing_core_property_fails_inspect(self):
        v = make_vertex_array(50, seed=1)
        names = [n for n in v.dtype.names if n != "opacity"]
        trimmed = np.empty(len(v), dtype=np.dtype([(n, v.dtype[n])
                                                   for n in names]))
        for n in names:
            trimmed[n] = v[n]
        src2 = write_ply(self.dir / "noopacity.ply", trimmed, sh_degree=3)
        res = run_pipeline(src2, self.dir, quiet=True)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error_kind"], "stage")
        self.assertEqual(res["error_stage"], "inspect")
        self.assertIn("missing core properties", res["error"])
        self.assertIn("opacity", res["error"])

    def test_empty_ply_fails_inspect(self):
        empty = np.empty(0, dtype=make_vertex_array(1, seed=4).dtype)
        src2 = write_ply(self.dir / "empty.ply", empty, sh_degree=3)
        res = run_pipeline(src2, self.dir, quiet=True)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error_kind"], "stage")
        self.assertEqual(res["error_stage"], "inspect")
        self.assertIn("no Gaussians", res["error"])
        self.assertTrue((self.dir / "empty_pipeline.json").is_file())

    def test_extra_properties_are_preserved(self):
        # a PLY with unknown extras (nx/ny/nz) must prune fine and keep them
        base = make_vertex_array(300, seed=6, sh_degree=0)
        dt = np.dtype([(n, base.dtype[n]) for n in base.dtype.names[:6]]
                      + [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
                      + [(n, base.dtype[n]) for n in base.dtype.names[6:]])
        v = np.empty(len(base), dtype=dt)
        for n in base.dtype.names:
            v[n] = base[n]
        for n in ("nx", "ny", "nz"):
            v[n] = np.float32(0.0)
        src2 = write_ply(self.dir / "extras.ply", v, sh_degree=0)
        res = run_pipeline(src2, self.dir, quiet=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        out = PlyData.read(str(self.dir / "extras_smart.ply"))["vertex"].data
        self.assertIn("nx", out.dtype.names)
        self.assertIn("nz", out.dtype.names)
        self.assertEqual(len(out), res["counts"]["output"])

    def test_sh_degree_equal_to_input_is_skipped(self):
        res = run_pipeline(self.src, self.dir, sh_degree=3, quiet=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        sh_stage = next(s for s in res["stages"]
                        if s["name"] == "sh_truncate")
        self.assertEqual(sh_stage["status"], "skipped")
        self.assertIn("already SH degree 3", sh_stage["reason"])
        self.assertFalse((self.dir / "scene_smart_shd3.ply").exists())
        self.assertEqual(res["outputs"]["sh_ply"], None)
        self.assertEqual(res["outputs"]["final_ply"],
                         str(self.dir / "scene_smart.ply"))

    def test_sh_degree_higher_than_input_fails_and_cleans_up(self):
        v2 = make_vertex_array(200, seed=2, sh_degree=2)
        src2 = write_ply(self.dir / "deg2.ply", v2, sh_degree=2)
        res = run_pipeline(src2, self.dir, sh_degree=3, quiet=True)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error_kind"], "stage")
        self.assertEqual(res["error_stage"], "sh_truncate")
        self.assertIn("cannot truncate", res["error"])
        # partial outputs of the failed run are removed, report remains
        self.assertFalse((self.dir / "deg2_smart.ply").exists())
        self.assertTrue((self.dir / "deg2_pipeline.json").is_file())

    def test_no_prune_no_sh_no_route_b(self):
        res = run_pipeline(self.src, self.dir, prune=False, quiet=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        skipped = {s["name"]: s["status"] for s in res["stages"]}
        self.assertEqual(skipped["smart_prune"], "skipped")
        self.assertEqual(skipped["sh_truncate"], "skipped")
        self.assertEqual(skipped["route_b"], "skipped")
        self.assertIsNone(res["outputs"]["final_ply"])
        self.assertEqual(res["counts"]["output"], 400)
        self.assertEqual(res["ratios"], {})
        self.assertTrue(any("smart_prune skipped" in w
                            for w in res["warnings"]))
        self.assertTrue((self.dir / "scene_pipeline.json").is_file())
        self.assertFalse((self.dir / "scene_smart.ply").exists())

    def test_route_b_without_prune_names_fall_back_to_input_stem(self):
        res = run_pipeline(self.src, self.dir, prune=False, route_b=True,
                           quiet=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        self.assertEqual(res["outputs"]["final_ply"], None)
        self.assertEqual(res["outputs"]["npz"], str(self.dir / "scene.npz"))
        self.assertEqual(res["outputs"]["q_back_ply"],
                         str(self.dir / "scene_qback.ply"))
        self.assertGreater(res["ratios"]["npz_vs_input"], 1.0)
        self.assertEqual(res["counts"]["output"], 400)

    def test_project_folder_convention(self):
        project = self.dir / "proj"
        (project / "scenes" / "original").mkdir(parents=True)
        (project / "project.json").write_text(
            json.dumps({"name": "proj", "version": 1}), encoding="utf-8")
        res = run_pipeline(self.src, project, quiet=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        self.assertEqual(res["project"], str(project))
        optimized = project / "scenes" / "optimized" / "scene_smart.ply"
        report = project / "results" / "scene_pipeline.json"
        csv_log = project / "results" / "pipeline_runs.csv"
        self.assertTrue(optimized.is_file())
        self.assertTrue(report.is_file())
        self.assertTrue(csv_log.is_file())
        self.assertEqual(res["outputs"]["final_ply"], str(optimized))
        self.assertEqual(res["outputs"]["report_json"], str(report))
        # nothing written outside the project
        self.assertFalse((project / "scene_smart.ply").exists())
        self.assertFalse((project / "scene_pipeline.json").exists())

    def test_output_colliding_with_input_is_rejected(self):
        # a report name equal to the input file would overwrite the input
        res = run_pipeline(self.src, self.dir, report_name="scene.ply",
                           quiet=True)
        self.assertEqual(res["error_kind"], "bad_input")
        self.assertEqual(res["exit_code"], 2)
        self.assertIn("overwrite the input", res["error"])
        self.assertEqual(self.src.read_bytes(), self.original_bytes)


class TestRouteBPresetPinning(PipelineTestCase):
    """Pin the Route B preset behind every published .npz number.

    The 10.26 MB (evaluation run) vs 7.41 MB (pipeline smoke run) sizes
    came from the same preset on DIFFERENT inputs: the evaluation run
    encoded the smart-pruned SH-degree-3 PLY (45 f_rest), the pipeline
    smoke run encoded the post-truncation SH-degree-2 PLY (24 f_rest)
    because pipeline Route B encodes final_ply. Same code, same bits,
    different input SH degree.
    """

    def test_pipeline_uses_core_default_opts(self):
        v = make_vertex_array(300, seed=12, sh_degree=3)
        src = write_ply(self.dir / "scene.ply", v, sh_degree=3)
        res = run_pipeline(src, self.dir, prune=False, route_b=True,
                           quiet=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        self.assertEqual(res["route_b"]["settings_bits"],
                         dict(route_b_core.DEFAULT_OPTS))

    def test_same_input_same_bytes_as_direct_encode(self):
        # pipeline route_b and core.encode_ply must agree byte for byte
        # on the same input with the default preset
        v = make_vertex_array(300, seed=13, sh_degree=3)
        src = write_ply(self.dir / "scene.ply", v, sh_degree=3)
        direct = self.dir / "direct.npz"
        route_b_core.encode_ply(src, direct)
        res = run_pipeline(src, self.dir, prune=False, route_b=True,
                           quiet=True)
        self.assertEqual(res["status"], "ok", res.get("error"))
        pipe_npz = Path(res["outputs"]["npz"])
        self.assertEqual(direct.stat().st_size, pipe_npz.stat().st_size)
        self.assertEqual(direct.read_bytes(), pipe_npz.read_bytes())

    def test_npz_size_depends_on_input_sh_degree(self):
        # same Gaussians, same preset: degree-3 input archives larger
        # than its degree-2 truncation (the 10.26 vs 7.41 MB cause)
        v = make_vertex_array(600, seed=14, sh_degree=3)
        src3 = write_ply(self.dir / "d3.ply", v, sh_degree=3)
        src2 = self.dir / "d2.ply"
        sh_truncate.truncate(src3, src2, 2)
        npz3 = self.dir / "d3.npz"
        npz2 = self.dir / "d2.npz"
        route_b_core.encode_ply(src3, npz3)
        route_b_core.encode_ply(src2, npz2)
        self.assertGreater(npz3.stat().st_size, npz2.stat().st_size)
        # f_rest dominates: 45 vs 24 float32 columns per Gaussian
        rest3 = sum(1 for n in v.dtype.names if n.startswith("f_rest_"))
        self.assertEqual(rest3, 45)


class TestPipelineCLI(PipelineTestCase):
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "optimization.pipeline", *args],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=300)

    def test_cli_exit_2_missing_input(self):
        r = self._run("--input", str(self.dir / "missing.ply"),
                      "--output-dir", str(self.dir))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("input not found", r.stderr + r.stdout)
        self.assertNotIn("Traceback", r.stderr)

    def test_cli_exit_3_bad_ply(self):
        bad = self.dir / "bad.ply"
        bad.write_text("nope\n", encoding="utf-8")
        r = self._run("--input", str(bad), "--output-dir", str(self.dir),
                      "--quiet")
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertTrue((self.dir / "bad_pipeline.json").is_file())
        self.assertNotIn("Traceback", r.stderr)

    def test_cli_non_ply_extension_clean_error(self):
        txt = self.dir / "note.txt"
        txt.write_text("hello", encoding="utf-8")
        r = self._run("--input", str(txt), "--output-dir", str(self.dir),
                      "--quiet")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("ERROR", r.stdout + r.stderr)

    def test_cli_existing_outputs_exit_2_no_traceback(self):
        src = write_ply(self.dir / "scene.ply",
                        make_vertex_array(200, seed=7), sh_degree=0)
        r1 = self._run("--input", str(src), "--output-dir", str(self.dir),
                       "--quiet")
        self.assertEqual(r1.returncode, 0, r1.stdout + r1.stderr)
        r2 = self._run("--input", str(src), "--output-dir", str(self.dir),
                       "--quiet")
        self.assertEqual(r2.returncode, 2, r2.stdout + r2.stderr)
        self.assertIn("refusing to overwrite", r2.stdout + r2.stderr)
        self.assertNotIn("Traceback", r2.stderr)

    def test_cli_exit_0_full_run(self):
        src = write_ply(self.dir / "scene.ply",
                        make_vertex_array(300, seed=3), sh_degree=3)
        r = self._run("--input", str(src), "--output-dir", str(self.dir),
                      "--sh-degree", "1", "--route-b", "--quiet")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Pipeline:  OK", r.stdout)
        self.assertTrue((self.dir / "scene_smart_shd1_qback.ply").is_file())
        self.assertTrue((self.dir / "scene_pipeline.json").is_file())


if __name__ == "__main__":
    unittest.main()
