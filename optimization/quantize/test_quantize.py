"""Unit tests for Route A/B/C pure functions.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest optimization.quantize.test_quantize -v
"""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from optimization.quantize import core, sh_truncate
from optimization.pruning.smart_prune import renormalize_quaternions


class TestQuantizeRoundtrip(unittest.TestCase):
    def test_uint8_roundtrip_within_step(self):
        rng = np.random.default_rng(0)
        v = rng.uniform(-3.0, 7.0, size=1000)
        q, lo, hi = core.quantize_values(v, 8)
        d = core.dequantize_values(q, lo, hi, 8)
        step = (hi - lo) / 255.0
        self.assertLessEqual(np.abs(v - d).max(), step / 2 * (1 + 1e-5) + 1e-12)
        self.assertEqual(q.dtype, np.uint8)

    def test_uint16_roundtrip_within_step(self):
        rng = np.random.default_rng(1)
        v = rng.normal(size=5000)
        q, lo, hi = core.quantize_values(v, 16)
        d = core.dequantize_values(q, lo, hi, 16)
        step = (hi - lo) / 65535.0
        self.assertLessEqual(np.abs(v - d).max(), step / 2 * (1 + 1e-5) + 1e-12)
        self.assertEqual(q.dtype, np.uint16)

    def test_constant_array(self):
        v = np.full(50, 4.25)
        q, lo, hi = core.quantize_values(v, 8)
        d = core.dequantize_values(q, lo, hi, 8)
        self.assertEqual(lo, hi)
        np.testing.assert_array_equal(d, v)

    def test_nonfinite_rejected(self):
        v = np.array([1.0, np.nan])
        with self.assertRaises(ValueError):
            core.quantize_values(v, 8)

    def test_bits_routing(self):
        opts = dict(core.DEFAULT_OPTS)
        self.assertEqual(core.bits_for_property("x", opts), 16)
        self.assertEqual(core.bits_for_property("opacity", opts), 16)
        self.assertEqual(core.bits_for_property("rot_2", opts), 8)
        self.assertEqual(core.bits_for_property("scale_1", opts), 16)
        self.assertEqual(core.bits_for_property("f_dc_0", opts), 16)
        self.assertEqual(core.bits_for_property("f_rest_44", opts), 8)
        with self.assertRaises(ValueError):
            core.bits_for_property("nx", opts)

    def test_validate_opts_rejects_bad_bits(self):
        with self.assertRaises(ValueError):
            core.validate_opts({**core.DEFAULT_OPTS, "rest": 12})


class TestQuaternionRoundtrip(unittest.TestCase):
    def test_decode_is_unit(self):
        rng = np.random.default_rng(2)
        q = rng.normal(size=(200, 4))
        quant, lo, hi, qn = core.quantize_rotations(q, 8)
        dec = core.dequantize_rotations(quant, lo, hi, 8)
        norms = np.linalg.norm(dec, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_component_error_within_step(self):
        rng = np.random.default_rng(3)
        q = renormalize_quaternions(rng.normal(size=(200, 4)))
        quant, lo, hi, qn = core.quantize_rotations(q, 8)
        dec_raw = core.dequantize_values(quant, lo, hi, 8)
        step = 2.0 / 255.0
        self.assertLessEqual(np.abs(qn - dec_raw).max(),
                             step / 2 * (1 + 1e-5) + 1e-12)

    def test_renormalize_helper(self):
        out = renormalize_quaternions(np.array([[0.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(out[0], [1, 0, 0, 0], atol=1e-12)


class TestSHTruncation(unittest.TestCase):
    def test_rest_count_for_degree(self):
        self.assertEqual(sh_truncate.REST_COUNT_FROM_DEG[0], 0)
        self.assertEqual(sh_truncate.REST_COUNT_FROM_DEG[1], 9)
        self.assertEqual(sh_truncate.REST_COUNT_FROM_DEG[2], 24)
        self.assertEqual(sh_truncate.REST_COUNT_FROM_DEG[3], 45)

    def test_keep_indices_degree3_to_2(self):
        keep = sh_truncate.rest_keep_old_indices(45, 2)
        self.assertEqual(keep, list(range(0, 8)) + list(range(15, 23))
                         + list(range(30, 38)))

    def test_keep_indices_degree3_to_1(self):
        keep = sh_truncate.rest_keep_old_indices(45, 1)
        self.assertEqual(keep, [0, 1, 2, 15, 16, 17, 30, 31, 32])

    def test_keep_indices_degree3_to_0(self):
        self.assertEqual(sh_truncate.rest_keep_old_indices(45, 0), [])

    def test_keep_indices_degree2_input_to_1(self):
        # degree-2 file: channel blocks of 8
        keep = sh_truncate.rest_keep_old_indices(24, 1)
        self.assertEqual(keep, [0, 1, 2, 8, 9, 10, 16, 17, 18])

    def test_cannot_raise_degree(self):
        with self.assertRaises(ValueError):
            sh_truncate.rest_keep_old_indices(24, 3)

    def test_update_sh_comment_replaces_in_place(self):
        c = ["Exported from Brush", "Vertical axis: y", "SH degree: 3"]
        out = sh_truncate.update_sh_comment(c, 1)
        self.assertEqual(out, ["Exported from Brush", "Vertical axis: y",
                               "SH degree: 1"])

    def test_update_sh_comment_appends_when_missing(self):
        out = sh_truncate.update_sh_comment(["only comment"], 2)
        self.assertEqual(out, ["only comment", "SH degree: 2"])

    def test_update_sh_comment_idempotent_order(self):
        c = ["a", "SH degree: 3", "b"]
        out = sh_truncate.update_sh_comment(c, 0)
        self.assertEqual(out, ["a", "SH degree: 0", "b"])

    def test_detect_layout_channel_major(self):
        # three similar blocks of 15 with small noise
        rng = np.random.default_rng(4)
        base = rng.uniform(0.003, 0.01, size=15)
        e = np.concatenate([base, base * 1.02, base * 0.98])
        det = sh_truncate.detect_layout(e)
        self.assertEqual(det["layout"], "channel_major")

    def test_detect_layout_coeff_interleaved(self):
        rng = np.random.default_rng(5)
        trips = rng.uniform(0.003, 0.01, size=15)
        e = np.repeat(trips, 3)  # same value across each channel triple
        det = sh_truncate.detect_layout(e)
        self.assertEqual(det["layout"], "coeff_interleaved")

    def test_band_means(self):
        b = sh_truncate.band_means(np.arange(15, dtype=np.float64))
        self.assertEqual(b, (1.0, 5.0, 11.0))

    def test_build_output_names_order(self):
        orig = ["x", "f_rest_10", "f_rest_0", "opacity", "y"]
        new = ["f_rest_0", "f_rest_1"]
        out = sh_truncate.build_output_names(orig, new)
        self.assertEqual(out, ["x", "f_rest_0", "f_rest_1", "opacity", "y"])


class TestGzipRouteC(unittest.TestCase):
    def test_gzip_byte_identical(self):
        data = bytes(np.random.default_rng(6).integers(0, 256, size=100000))
        gz = gzip.compress(data, compresslevel=6, mtime=0)
        self.assertEqual(gzip.decompress(gz), data)
        self.assertLess(len(gz), len(data))  # random-ish data still shrinks? no!
        # note: pure random data does not shrink; use structured data instead


class TestGzipRouteCStructured(unittest.TestCase):
    def test_gzip_shrinks_and_roundtrips(self):
        # float32 structured data (like a PLY) compresses well
        rng = np.random.default_rng(7)
        arr = rng.normal(size=50000).astype(np.float32)
        arr = np.repeat(arr, 12)  # 59-property-like redundancy
        data = arr.tobytes()
        gz = gzip.compress(data, compresslevel=6, mtime=0)
        self.assertEqual(gzip.decompress(gz), data)
        self.assertLess(len(gz), len(data))


class TestEncodeDecodeEndToEnd(unittest.TestCase):
    def _write_tiny_ply(self, path, n=200):
        rng = np.random.default_rng(8)
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
                       ("f_rest_0", "<f4"), ("f_rest_1", "<f4"),
                       ("f_rest_2", "<f4"), ("opacity", "<f4"),
                       ("rot_0", "<f4"), ("rot_1", "<f4"),
                       ("rot_2", "<f4"), ("rot_3", "<f4"),
                       ("scale_0", "<f4"), ("scale_1", "<f4"),
                       ("scale_2", "<f4")])
        v = np.zeros(n, dtype=dt)
        for name in dt.names:
            v[name] = rng.normal(size=n).astype("<f4")
        # unit quaternions like a real export
        q = renormalize_quaternions(np.stack([v[k] for k in
                                              ("rot_0", "rot_1", "rot_2",
                                               "rot_3")], 1))
        for i, k in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
            v[k] = q[:, i].astype("<f4")
        comments = ["Exported from Brush", "Vertical axis: y", "SH degree: 0"]
        PlyData([PlyElement.describe(v, "vertex")],
                text=False, byte_order="<", comments=comments).write(str(path))
        return v, comments

    def test_encode_decode_preserves_format_and_errors_small(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            npz = Path(td) / "out.npz"
            back = Path(td) / "back.ply"
            v, comments = self._write_tiny_ply(src)

            stats = core.encode_ply(src, npz)
            self.assertTrue(npz.is_file())
            rep = stats["report"]
            self.assertTrue(rep["finite"])
            self.assertTrue(rep["within_step_bound"])
            self.assertLess(rep["position"]["max_rel_to_diagonal"], 1e-3)
            self.assertAlmostEqual(rep["quaternion"]["post_renorm_norm_min"],
                                   1.0, places=12)
            self.assertAlmostEqual(rep["quaternion"]["post_renorm_norm_max"],
                                   1.0, places=12)

            info = core.decode_ply(npz, back)
            out = PlyData.read(str(back), mmap=None)
            self.assertEqual(list(out["vertex"].data.dtype.names),
                             list(v.dtype.names))
            self.assertEqual(list(out.comments), comments)
            self.assertEqual(len(out["vertex"].data), len(v))
            # decoded PLY must be finite and quaternion rows unit
            ov = out["vertex"].data
            self.assertTrue(all(np.isfinite(ov[k]).all()
                                for k in ov.dtype.names))
            qn = np.linalg.norm(np.stack([ov[f"rot_{i}"] for i in range(4)], 1),
                                axis=1)
            np.testing.assert_allclose(qn, 1.0, atol=1e-6)
            # end-to-end error still within step bounds
            for n in v.dtype.names:
                err = np.abs(v[n].astype(np.float64)
                             - ov[n].astype(np.float64)).max()
                p = stats["plan"][n]
                step = ((p["max"] - p["min"]) / ((1 << p["bits"]) - 1)
                        if p["max"] > p["min"] else 0.0)
                if n.startswith("rot_"):
                    delta = step / 2.0
                    bound = 3.0 * delta / (1.0 - 2.0 * delta) + 1e-9
                else:
                    bound = step / 2 * (1 + 1e-4) + 1e-6
                self.assertLessEqual(err, bound, msg=n)

    def test_archive_meta_contents(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ply"
            npz = Path(td) / "out.npz"
            self._write_tiny_ply(src, n=10)
            core.encode_ply(src, npz)
            with np.load(str(npz), allow_pickle=False) as z:
                meta = json.loads(bytes(z[core.META_KEY]).decode("utf-8"))
            self.assertEqual(meta["format"], core.FORMAT)
            self.assertEqual(meta["count"], 10)
            self.assertEqual(meta["source"], "in.ply")
            self.assertEqual(len(meta["properties"]), 17)
            self.assertEqual(meta["comments"],
                             ["Exported from Brush", "Vertical axis: y",
                              "SH degree: 0"])
            self.assertEqual(meta["plan"]["rot_0"]["bits"], 8)
            self.assertEqual(meta["plan"]["x"]["bits"], 16)


if __name__ == "__main__":
    unittest.main()
