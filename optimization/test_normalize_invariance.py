"""End-to-end tests for mirror clip, warning propagation, and normalize
invariance under rigid + uniform-scale transforms.

Scene design (floor_scene): a regular-grid floor at y=0, an odd 7^3
bright lattice subject centred (5, 4, 3), sparse bright fillers below
the subject, and a dim mirror cluster at y=-4.  The geometry is chosen
so subject detection is EXACTLY covariant under any rigid +
uniform-scale transform: the p25 k-NN cut lands in the gap between the
0.495 and 0.606 distance classes (candidates = 343 subject + 156
fillers = 499, percentile index 124.5, i.e. the last interior value),
so the dense set is the 125 interior lattice points in every frame and
the per-axis weighted median is the exact lattice centre.  A spurious
plane cutting the subject would break that symmetry, so the floor must
be the dominant RANSAC plane in every frame (fixed spacing, constant
opacity, far above the subject's k-NN classes); floor inliers are then
excluded from subject candidates in every frame.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest optimization.test_normalize_invariance -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from optimization.normalize.core import mat_to_quat, quat_mul
from optimization.pipeline import run_pipeline

REST_COUNTS = {0: 0, 1: 9, 2: 24, 3: 45}
SUBJECT_CENTRE = np.array([5.0, 4.0, 3.0])
SUBJECT_R90 = 1.05  # 3 * 0.35 * sqrt(3), p90 of the interior lattice
N_MIRROR = 90


def make_blank(n, sh_degree=3, seed=0):
    rng = np.random.default_rng(seed)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
    k = REST_COUNTS[sh_degree]
    fields += [(name, "<f4") for name in sorted(f"f_rest_{i}" for i in range(k))]
    fields += [("opacity", "<f4"),
               ("rot_0", "<f4"), ("rot_1", "<f4"),
               ("rot_2", "<f4"), ("rot_3", "<f4"),
               ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4")]
    v = np.zeros(n, dtype=np.dtype(fields))
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(4):
        v[f"rot_{i}"] = q[:, i].astype("<f4")
    v["opacity"] = rng.normal(-6.0, 1.0, n).astype("<f4")
    for i in range(3):
        v[f"f_dc_{i}"] = rng.normal(0.0, 0.3, n).astype("<f4")
        v[f"scale_{i}"] = rng.normal(-2.0, 0.3, n).astype("<f4")
    if k:
        cp = k // 3
        base = np.exp(-np.arange(cp) / 2.0)
        for c in range(3):
            for i in range(cp):
                v[f"f_rest_{c * cp + i}"] = rng.normal(
                    0.0, np.sqrt(base[i] * (1 + 0.05 * c)), n).astype("<f4")
    return v


def write_ply(path, v, sh_degree=3):
    ply = PlyData([PlyElement.describe(v, "vertex")], text=False,
                  byte_order="<",
                  comments=["Exported from Brush", "Vertical axis: y",
                            f"SH degree: {sh_degree}"])
    ply.write(str(path))
    return path


def floor_scene(n_side=20, n_sub_side=7, n_fill=156, n_mirror=N_MIRROR,
                seed=3):
    """Regular floor + odd subject lattice + sparse fillers + mirror.

    See the module docstring for why this layout makes subject
    detection exactly covariant under rigid + uniform-scale transforms.
    """
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side),
                             indexing="ij"), axis=-1).reshape(-1, 2)
    n_floor = len(g)
    n_sub = n_sub_side ** 3
    n = n_floor + n_sub + n_fill + n_mirror
    v = make_blank(n, 3, seed)
    # floor: regular grid spacing 1.0, y ~ 0, constant bright opacity
    v["x"][:n_floor] = ((g[:, 0] - (n_side - 1) / 2.0) * 1.0).astype("<f4")
    v["y"][:n_floor] = rng.normal(0.0, 0.05, n_floor).astype("<f4")
    v["z"][:n_floor] = ((g[:, 1] - (n_side - 1) / 2.0) * 1.0).astype("<f4")
    v["opacity"][:n_floor] = 1.0
    for i in range(3):
        v[f"scale_{i}"][:n_floor] = -1.0
    # subject: odd 7^3 lattice centred (5, 4, 3), step 0.35
    i0 = n_floor
    i1 = i0 + n_sub
    m = (n_sub_side - 1) // 2
    gg = np.stack(np.meshgrid(np.arange(n_sub_side), np.arange(n_sub_side),
                              np.arange(n_sub_side), indexing="ij"),
                  axis=-1).reshape(-1, 3)
    v["x"][i0:i1] = (SUBJECT_CENTRE[0] + (gg[:, 0] - m) * 0.35).astype("<f4")
    v["y"][i0:i1] = (SUBJECT_CENTRE[1] + (gg[:, 1] - m) * 0.35).astype("<f4")
    v["z"][i0:i1] = (SUBJECT_CENTRE[2] + (gg[:, 2] - m) * 0.35).astype("<f4")
    v["opacity"][i0:i1] = 2.0
    for i in range(3):
        v[f"scale_{i}"][i0:i1] = -0.7
    # fillers: sparse, bright, isotropic, strictly below the subject
    i2 = i1 + n_fill
    v["x"][i1:i2] = rng.uniform(-9.0, 9.0, n_fill).astype("<f4")
    v["y"][i1:i2] = rng.uniform(1.0, 2.4, n_fill).astype("<f4")
    v["z"][i1:i2] = rng.uniform(-9.0, 9.0, n_fill).astype("<f4")
    v["opacity"][i1:i2] = 1.0
    for i in range(3):
        v[f"scale_{i}"][i1:i2] = -1.0
    # mirror: dim cluster below floor at y=-4 (opacity below the scene
    # median, so it never enters plane/subject candidates)
    m2 = rng.normal(size=(n_mirror, 3)) * np.array([1.5, 1.0, 1.5])
    v["x"][i2:] = (5.0 + m2[:, 0]).astype("<f4")
    v["y"][i2:] = (-4.0 + m2[:, 1]).astype("<f4")
    v["z"][i2:] = (3.0 + m2[:, 2]).astype("<f4")
    v["opacity"][i2:] = 0.0
    for i in range(3):
        v[f"scale_{i}"][i2:] = -1.3
    return v


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def transform_scene(v, seed):
    """Random rotation + translation + uniform scale, float32 output."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    R = quat_to_matrix(q)
    t = rng.uniform(-30.0, 30.0, 3)
    a = float(rng.uniform(0.2, 5.0))
    out = v.copy()
    p = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    p2 = a * (p @ R.T) + t
    for i, ax in enumerate("xyz"):
        out[ax] = p2[:, i].astype("<f4")
    # uniform geometric scale: every log axis shifts by ln(a)
    for i in range(3):
        out[f"scale_{i}"] = (v[f"scale_{i}"].astype(np.float64)
                             + np.log(a)).astype("<f4")
    # world-rotate the orientations: q' = q_R (x) q  (rot_0 = w)
    qR = mat_to_quat(R)
    qv = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    q2 = quat_mul(np.repeat(qR[None, :], len(v), axis=0), qv)
    for i in range(4):
        out[f"rot_{i}"] = q2[:, i].astype("<f4")
    return out, R, t, a


def read_xyz(path):
    d = PlyData.read(str(path))["vertex"].data
    return (np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64),
            np.stack([d[f"rot_{i}"] for i in range(4)], axis=1)
            .astype(np.float64),
            np.stack([d[f"scale_{i}"] for i in range(3)], axis=1)
            .astype(np.float64),
            d)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestMirrorClipEndToEnd(TempDirTestCase):
    """mirror_clip + normalize + upright on the floor scene in one run."""

    def setUp(self):
        super().setUp()
        self.v = floor_scene()
        self.src = write_ply(self.dir / "floor.ply", self.v)
        self.res = run_pipeline(self.src, self.dir, mirror_clip=True,
                                normalize=True, upright=True, quiet=True)
        self.assertEqual(self.res["status"], "ok", self.res.get("error"))

    def test_clip_removes_mirror_only(self):
        clip = self.res["mirror_clip"]
        # nearly all of the 90 mirror points, none of the bright scene
        self.assertGreaterEqual(clip["removed_count"], 80)
        self.assertLessEqual(clip["removed_count"], 95)
        # the guard: the removed side carries almost no scene weight
        self.assertLess(clip["removed_weight_share"], 0.1)
        # raw fitted normal points +y; the minority side is below, so no
        # flip is needed and the kept side holds the subject
        self.assertFalse(clip["normal_flipped"])
        self.assertGreaterEqual(clip["plane_confidence"], 0.3)
        counts = self.res["counts"]
        # counts hold the number REMAINING after each stage
        self.assertEqual(counts["output"], counts["clipped"])
        self.assertEqual(clip["removed_count"],
                         counts["pruned"] - counts["clipped"])
        self.assertLessEqual(counts["pruned"], counts["input"])
        out_ply = self.res["outputs"]["final_ply"]
        self.assertIsNotNone(out_ply)
        out = PlyData.read(out_ply)["vertex"].data
        self.assertEqual(len(out), counts["output"])

    def test_warnings_reach_pipeline_result(self):
        warnings = self.res["warnings"]
        self.assertTrue(
            any(w.startswith("normalize") and "up confidence" in w
                for w in warnings),
            f"normalize warning missing from {warnings}")

    def test_subject_detected_exactly(self):
        plan = self.res["normalize"]["plan"]
        self.assertIn("subject density centre", plan["methods"]["center"])
        self.assertGreaterEqual(plan["confidence"]["center"], 0.5)
        sidecar = json.loads(
            Path(self.res["normalize"]["sidecar_path"]).read_text())
        np.testing.assert_allclose(sidecar["center"], SUBJECT_CENTRE,
                                   atol=1e-5)
        self.assertAlmostEqual(sidecar["s"], 1.0 / SUBJECT_R90, places=5)


class TestNormalizeInvariance(TempDirTestCase):
    """Normalized output must agree across rigid + uniform-scale frames."""

    # observed worst case across seeds: 7.6e-5 (seed 13); seeds below
    # stay under 6e-6, so 1e-4 keeps a 16x margin
    COORD_TOL = 1e-4
    SEEDS = (7, 1, 3)

    def setUp(self):
        super().setUp()
        self.v = floor_scene()
        self.src = write_ply(self.dir / "floor.ply", self.v)

    def test_detection_is_covariant(self):
        # analyse_scene level: the detected centre/r90 must transform
        # exactly with the scene, in every frame
        from optimization import analyze_scene
        r1 = analyze_scene.analyze(str(self.src))
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                tv, R, t, a = transform_scene(self.v, seed)
                tp = write_ply(self.dir / f"t{seed}.ply", tv)
                r2 = analyze_scene.analyze(str(tp))
                c1 = np.array(r1["subject"]["centre"])
                c2 = np.array(r2["subject"]["centre"])
                # frame 1 detects the exact lattice centre; frame 2 must
                # detect the same point moved by the transform
                np.testing.assert_allclose(c1, SUBJECT_CENTRE, atol=1e-5)
                expect = a * (R @ SUBJECT_CENTRE) + t
                self.assertLess(np.linalg.norm(c2 - expect), 1e-5)
                ratio = (r2["subject"]["radius_p90"]
                         / r1["subject"]["radius_p90"])
                self.assertLess(abs(ratio - a) / a, 1e-5)
                self.assertGreaterEqual(r1["subject"]["confidence"], 0.5)
                self.assertGreaterEqual(r2["subject"]["confidence"], 0.5)
                # the dominant plane must stay the floor in both frames
                # (a plane cutting the subject would break symmetry):
                # frame 1's normal is +y, frame 2's is R(+y)
                for rep, up in ((r1, np.array([0.0, 1.0, 0.0])),
                                (r2, R @ np.array([0.0, 1.0, 0.0]))):
                    n = np.array(rep["plane"]["normal"])
                    self.assertGreater(abs(float(n @ up)), 0.99)
                    self.assertGreaterEqual(rep["plane"]["confidence"], 0.3)

    def test_normalized_output_agrees_across_frames(self):
        r1 = run_pipeline(self.src, self.dir / "inv1", normalize=True,
                          prune=False, quiet=True)
        self.assertEqual(r1["status"], "ok", r1.get("error"))
        pa, qa, sa, da = read_xyz(self.dir / "inv1" / "floor_norm.ply")
        s1 = json.loads(
            (self.dir / "inv1" / "floor_norm.normalize.json").read_text())
        self.assertIn("subject density centre", s1["methods"]["center"])
        np.testing.assert_allclose(s1["center"], SUBJECT_CENTRE, atol=1e-5)
        self.assertGreaterEqual(s1["confidence"]["center"], 0.5)

        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                tv, R, t, a = transform_scene(self.v, seed)
                tp = write_ply(self.dir / f"t{seed}.ply", tv)
                r2 = run_pipeline(tp, self.dir / f"inv{seed}",
                                  normalize=True, prune=False, quiet=True)
                self.assertEqual(r2["status"], "ok", r2.get("error"))
                pb, qb, sb, db = read_xyz(
                    self.dir / f"inv{seed}" / f"t{seed}_norm.ply")
                s2 = json.loads(
                    (self.dir / f"inv{seed}" /
                     f"t{seed}_norm.normalize.json").read_text())

                # same subject detection in both frames: the method is
                # the subject centre and it moves with the transform
                self.assertIn("subject density centre",
                              s2["methods"]["center"])
                self.assertGreaterEqual(s2["confidence"]["center"], 0.5)

                # centre covariant: c2 = a R c1 + t
                c2 = np.array(s2["center"])
                expect = a * (R @ np.array(s1["center"])) + t
                self.assertLess(np.linalg.norm(c2 - expect), 1e-5)

                # scale covariant: s1 / s2 = a
                self.assertLess(abs(s1["s"] / s2["s"] - a) / a, 1e-5)

                # positions: normalizing frame 2 and rotating back by R
                # reproduces frame 1 (p_out = s (p_in - c))
                self.assertLess(np.abs(pa - pb @ R).max(), self.COORD_TOL)

                # orientations: local quats differ only by the world
                # rotation the transform applied
                qR = mat_to_quat(R)
                qRinv = qR * np.array([1.0, -1.0, -1.0, -1.0])
                pred = quat_mul(np.repeat(qRinv[None, :], len(qb), axis=0),
                                qb)
                dot = np.abs((qa * pred).sum(axis=1))
                self.assertLess((1.0 - dot).max(), 1e-5)

                # log-scales are axiswise equal (uniform scale keeps
                # anisotropy, so the median filter cannot reorder them)
                self.assertLess(np.abs(sa - sb).max(), 1e-5)

                # colour/opacity untouched (R = I: no SH rotation)
                for field in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity"):
                    np.testing.assert_array_equal(da[field], db[field])


if __name__ == "__main__":
    unittest.main()
