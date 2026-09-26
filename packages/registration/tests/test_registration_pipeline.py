from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from skimage.transform import AffineTransform as SkimageAffineTransform

import ufwbpp_registration.pipeline as registration_pipeline
from ufwbpp_registration.pipeline import (
    CalibrationPlan,
    calibrate_image,
    RegistrationConfig,
    warp_image,
)
from ufwbpp_registration import (
    estimate_stellar_scale_hints,
    normalize_quality_weights,
)
from ufwbpp_registration.pipeline import FrameAnalysis, StarCatalog


class RegistrationGeometryTests(unittest.TestCase):
    def test_stellar_scale_hint_removes_exposure_ratio_and_isolates_filters(
        self,
    ) -> None:
        shape = (512, 512)
        yy, xx = np.indices(shape, dtype=np.float64)
        points = np.asarray(
            [(x, y) for y in (80.0, 160.0, 240.0, 320.0, 400.0) for x in (80.0, 160.0, 240.0, 320.0, 400.0)],
            dtype=np.float64,
        )

        def pixels(exposure: float, background: float, outlier: bool = False) -> np.ndarray:
            image = np.full(shape, background, dtype=np.float64)
            image += exposure * 4.0 * np.exp(
                -((xx - 260.0) ** 2 + (yy - 250.0) ** 2) / (2.0 * 70.0**2)
            )
            for index, (x, y) in enumerate(points):
                amplitude = exposure * (80.0 + index)
                if outlier and index == 4:
                    amplitude *= 20.0
                image += amplitude * np.exp(
                    -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.1**2)
                )
            return image.astype(np.float32)

        def analysis(
            path: str,
            filter_name: str,
            exposure: float,
            image: np.ndarray,
        ) -> FrameAnalysis:
            count = len(points)
            return FrameAnalysis(
                path=path,
                filter_name=filter_name,
                preview=image,
                source_width=2048,
                source_height=2048,
                scale_x=4.0,
                scale_y=4.0,
                catalog=StarCatalog(
                    points=points,
                    flux=np.linspace(1000.0, 500.0, count) * exposure,
                    peak=np.linspace(100.0, 50.0, count) * exposure,
                    fwhm=np.full(count, 2.8),
                    background=0.0,
                    noise=1.0,
                    detected_count=count,
                ),
                read_seconds=0.0,
                calibration_seconds=0.0,
                detection_seconds=0.0,
                exposure_seconds=exposure,
            )

        analyses = (
            analysis("/r-reference.fits", "R", 30.0, pixels(30.0, 500.0)),
            analysis("/r-source.fits", "R", 60.0, pixels(60.0, 900.0, True)),
            analysis("/b-reference.fits", "B", 30.0, pixels(30.0, 700.0)),
        )
        transforms = tuple(
            SimpleNamespace(preview_matrix=np.eye(3, dtype=np.float64))
            for _ in analyses
        )
        estimates = estimate_stellar_scale_hints(
            analyses,
            transforms,
            (1.0, 0.9, 1.0),
        )

        self.assertEqual(estimates[0].status, "REFERENCE_IDENTITY")
        self.assertEqual(estimates[1].status, "STELLAR_SCALE_ACCEPTED")
        self.assertAlmostEqual(estimates[1].scale or 0.0, 1.0, delta=0.02)
        self.assertEqual(
            estimates[1].evidence["scaleDomain"],
            "post-linear-exposure-normalization",
        )
        self.assertEqual(estimates[1].evidence["exposureCorrection"], 2.0)
        self.assertEqual(estimates[2].reference_index, 2)
        self.assertEqual(estimates[2].scale, 1.0)

    def test_full_resolution_refinement_defaults_to_affine(self) -> None:
        config = RegistrationConfig()
        self.assertTrue(config.refine_full_centroids)
        self.assertEqual(config.full_transform_model, "affine")

    def test_full_resolution_refinement_rejects_unknown_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "full_transform_model"):
            RegistrationConfig(full_transform_model="polynomial")

    def test_full_resolution_affine_refinement_recovers_shear(self) -> None:
        reference_points = np.asarray(
            [(x, y) for y in (24.0, 48.0, 72.0, 96.0) for x in (24.0, 48.0, 72.0, 96.0)],
            dtype=np.float64,
        )
        expected = np.asarray(
            [[1.002, 0.018, 1.2], [-0.012, 0.997, -0.8], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        homogeneous = np.column_stack(
            (reference_points, np.ones(reference_points.shape[0], dtype=np.float64))
        )
        source_points = (np.linalg.inv(expected) @ homogeneous.T).T[:, :2]

        def image(points: np.ndarray) -> np.ndarray:
            yy, xx = np.indices((128, 128), dtype=np.float64)
            result = np.zeros((128, 128), dtype=np.float64)
            for x, y in points:
                result += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.2**2))
            return result.astype(np.float32)

        def analysis(path: str, points: np.ndarray, pixels: np.ndarray) -> FrameAnalysis:
            count = points.shape[0]
            return FrameAnalysis(
                path=path,
                filter_name="R",
                preview=pixels,
                source_width=128,
                source_height=128,
                scale_x=1.0,
                scale_y=1.0,
                catalog=StarCatalog(
                    points=points,
                    flux=np.linspace(1000.0, 500.0, count),
                    peak=np.linspace(100.0, 50.0, count),
                    fwhm=np.full(count, 2.8),
                    background=0.0,
                    noise=0.01,
                    detected_count=count,
                ),
                read_seconds=0.0,
                calibration_seconds=0.0,
                detection_seconds=0.0,
            )

        source_pixels = image(source_points)
        reference_pixels = image(reference_points)
        source = analysis("/source.fits", source_points, source_pixels)
        reference = analysis("/reference.fits", reference_points, reference_pixels)
        coarse = SkimageAffineTransform(translation=(1.0, -1.0))
        initial = registration_pipeline._result_from_model(
            source,
            reference,
            coarse,
            np.full(reference_points.shape[0], 1.0),
            reference_points.shape[0],
            RegistrationConfig(refine_full_centroids=False),
            started=0.0,
        )
        refined = registration_pipeline._refine_full_resolution(
            initial,
            source,
            reference,
            source_pixels,
            reference_pixels,
            RegistrationConfig(
                full_transform_model="affine",
                full_centroid_radius_px=5,
                full_residual_threshold_px=0.35,
            ),
        )
        self.assertEqual(refined.transform_model, "affine-full-centroid")
        self.assertGreaterEqual(refined.full_refine_inliers, 12)
        np.testing.assert_allclose(refined.full_matrix, expected, rtol=0.0, atol=0.02)
        np.testing.assert_allclose(refined.full_matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-12)

    def test_full_resolution_affine_rejects_spatially_clustered_control_points(
        self,
    ) -> None:
        points = np.asarray(
            [(x, y) for y in (35.0, 42.0, 49.0, 56.0) for x in (35.0, 42.0, 49.0, 56.0)],
            dtype=np.float64,
        )
        yy, xx = np.indices((128, 128), dtype=np.float64)
        pixels = np.zeros((128, 128), dtype=np.float64)
        for x, y in points:
            pixels += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.0**2))
        count = points.shape[0]

        def analysis(path: str) -> FrameAnalysis:
            return FrameAnalysis(
                path=path,
                filter_name="R",
                preview=pixels.astype(np.float32),
                source_width=128,
                source_height=128,
                scale_x=1.0,
                scale_y=1.0,
                catalog=StarCatalog(
                    points=points,
                    flux=np.linspace(1000.0, 500.0, count),
                    peak=np.linspace(100.0, 50.0, count),
                    fwhm=np.full(count, 2.8),
                    background=0.0,
                    noise=0.01,
                    detected_count=count,
                ),
                read_seconds=0.0,
                calibration_seconds=0.0,
                detection_seconds=0.0,
            )

        source = analysis("/source.fits")
        reference = analysis("/reference.fits")
        initial = registration_pipeline._result_from_model(
            source,
            reference,
            SkimageAffineTransform(),
            np.zeros(count),
            count,
            RegistrationConfig(refine_full_centroids=False),
            started=0.0,
        )
        refined = registration_pipeline._refine_full_resolution(
            initial,
            source,
            reference,
            pixels.astype(np.float32),
            pixels.astype(np.float32),
            RegistrationConfig(full_centroid_radius_px=3),
        )

        self.assertEqual(refined.transform_model, "similarity")
        self.assertEqual(refined.full_refine_evidence["status"], "REJECTED")
        self.assertEqual(
            refined.full_refine_evidence["reason"], "FULL_MODEL_GATE_FAILED"
        )
        self.assertIn(
            "SPATIAL_SPAN_LOW", refined.full_refine_evidence["gateReasons"]
        )

    def test_warp_uses_source_to_reference_contract(self) -> None:
        image = np.zeros((20, 30), dtype=np.float32)
        image[8, 9] = 1
        translated = np.asarray([[1, 0, 4], [0, 1, 3], [0, 0, 1]], dtype=float)
        warped = warp_image(image, translated, image.shape, order=0, cval=0)
        y, x = np.unravel_index(int(np.argmax(warped)), warped.shape)
        self.assertEqual((x, y), (13, 11))

    def test_projective_warp_uses_same_contract(self) -> None:
        image = np.zeros((30, 40), dtype=np.float32)
        image[8, 10] = 1
        projective = np.asarray(
            [[1, 0, 2], [0, 1, 1], [0.001, 0, 1]], dtype=float
        )
        warped = warp_image(image, projective, image.shape, order=0, cval=0)
        y, x = np.unravel_index(int(np.argmax(warped)), warped.shape)
        self.assertLessEqual(abs(x - 12), 1)
        self.assertLessEqual(abs(y - 9), 1)

    def test_bias_dark_and_flat_calibration_contract(self) -> None:
        light = np.full((4, 4), 110, dtype=np.float32)
        bias = np.full((4, 4), 10, dtype=np.float32)
        dark = np.full((4, 4), 14, dtype=np.float32)
        flat = np.full((4, 4), 2, dtype=np.float32)
        result = calibrate_image(
            light,
            CalibrationPlan(dark_includes_bias=True),
            filter_name="R",
            bias=bias,
            dark=dark,
            flat=flat,
        )
        np.testing.assert_array_equal(result, np.full((4, 4), 96, dtype=np.float32))

    def test_a_light_with_its_own_masters_ignores_the_filter_lookup(self) -> None:
        import tempfile
        from pathlib import Path

        from astropy.io import fits

        from ufwbpp_registration.pipeline import DetectionConfig, LightMasters, _read_calibrated_preview

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, pixels: np.ndarray, imagetyp: str) -> str:
                header = fits.Header()
                header["IMAGETYP"] = imagetyp
                header["FILTER"] = "L"
                header["EXPTIME"] = 10.0
                fits.PrimaryHDU(pixels.astype(np.float32), header=header).writeto(root / name)
                return str(root / name)

            light = write("light.fits", np.full((32, 32), 110.0), "Light Frame")
            bias = write("bias.fits", np.full((32, 32), 10.0), "Master Bias")
            filter_flat = write("flat_filter.fits", np.full((32, 32), 2.0), "Master Flat")
            night_flat_pixels = np.tile(np.linspace(1.0, 3.0, 32), (32, 1))
            night_flat = write("flat_night.fits", night_flat_pixels, "Master Flat")
            plan = CalibrationPlan(
                bias_path=bias,
                flat_paths={"L": filter_flat},
                light_masters={light: LightMasters(bias_path=bias, flat_path=night_flat)},
            )
            _, calibrated, _ = _read_calibrated_preview(light, DetectionConfig(), plan, {}, None)
            expected = 100.0 / (night_flat_pixels / np.median(night_flat_pixels))
            np.testing.assert_allclose(calibrated, expected.astype(np.float32), rtol=1e-6)
            without = CalibrationPlan(bias_path=bias, flat_paths={"L": filter_flat})
            _, calibrated, _ = _read_calibrated_preview(light, DetectionConfig(), without, {}, None)
            np.testing.assert_allclose(calibrated, np.full((32, 32), 100.0, dtype=np.float32))

    def test_normalized_additive_masters_are_scaled_into_integer_light_domain(self) -> None:
        light = np.full((4, 4), 631.0, dtype=np.float32)
        bias = np.full((4, 4), 481.0 / 65535.0, dtype=np.float32)
        dark = np.full((4, 4), 501.0 / 65535.0, dtype=np.float32)
        result = calibrate_image(
            light,
            CalibrationPlan(
                dark_includes_bias=True,
                bias_application_scale=65535.0,
                dark_scale=65535.0,
            ),
            filter_name="R",
            bias=bias,
            dark=dark,
        )
        np.testing.assert_allclose(
            result,
            np.full((4, 4), 130.0, dtype=np.float32),
            rtol=0.0,
            atol=2e-5,
        )

    def test_native_quality_weight_is_group_median_normalized(self) -> None:
        def analysis(filter_name: str, flux: float) -> FrameAnalysis:
            return FrameAnalysis(
                path=f"/{filter_name}-{flux}.fits",
                filter_name=filter_name,
                preview=np.zeros((16, 16), dtype=np.float32),
                source_width=16,
                source_height=16,
                scale_x=1,
                scale_y=1,
                catalog=StarCatalog(
                    points=np.asarray([[1, 1], [2, 2]], dtype=np.float64),
                    flux=np.asarray([flux, flux], dtype=np.float64),
                    peak=np.asarray([flux * flux, flux * flux], dtype=np.float64),
                    fwhm=np.asarray([2, 2], dtype=np.float64),
                    background=0,
                    noise=1,
                    detected_count=2,
                ),
                read_seconds=0,
                calibration_seconds=0,
                detection_seconds=0,
            )

        weights = normalize_quality_weights(
            [analysis("R", 1), analysis("R", 3), analysis("G", 4)]
        )
        np.testing.assert_allclose(weights, (0.5, 1.5, 1.0))


if __name__ == "__main__":
    unittest.main()


class ReferenceChoiceTests(unittest.TestCase):
    @staticmethod
    def _analysis(path: str, count: int, fwhm: float) -> FrameAnalysis:
        points = np.column_stack([np.linspace(10.0, 100.0, count), np.linspace(10.0, 100.0, count)])
        return FrameAnalysis(
            path=path,
            filter_name="L",
            preview=np.zeros((128, 128), dtype=np.float32),
            source_width=128,
            source_height=128,
            scale_x=1.0,
            scale_y=1.0,
            catalog=StarCatalog(
                points=points,
                flux=np.full(count, 500.0),
                peak=np.full(count, 50.0),
                fwhm=np.full(count, fwhm),
                background=0.0,
                noise=1.0,
                detected_count=count,
            ),
            read_seconds=0.0,
            calibration_seconds=0.0,
            detection_seconds=0.0,
            exposure_seconds=300.0,
        )

    def test_candidates_keep_a_noise_blob_frame_from_anchoring_the_registration(self) -> None:
        # A cloud-covered frame: 500 sub-pixel noise detections score 500 / 0.25,
        # far above real frames with 300 stars of 2.6 px FWHM.
        analyses = (
            self._analysis("/clear-a.fits", 300, 2.6),
            self._analysis("/clear-b.fits", 320, 2.5),
            self._analysis("/cloud.fits", 500, 0.5),
        )
        self.assertEqual(registration_pipeline.choose_reference(analyses), 2)
        self.assertEqual(registration_pipeline.choose_reference(analyses, [0, 1]), 1)
        # An empty candidate list means "no restriction", never "no frames".
        self.assertEqual(registration_pipeline.choose_reference(analyses, []), 2)
        with self.assertRaises(IndexError):
            registration_pipeline.choose_reference(analyses, [3])


class LocalCentroidReferenceTests(unittest.TestCase):
    """The gathered centroid refinement equals the one-star-at-a-time loop."""

    @staticmethod
    def _reference(image, approximate, radius):
        import math

        height, width = image.shape
        refined, retained = [], []
        size = 2 * radius + 1
        for index, (x, y) in enumerate(approximate):
            center_x = int(round(float(x)))
            center_y = int(round(float(y)))
            x0, x1 = center_x - radius, center_x + radius + 1
            y0, y1 = center_y - radius, center_y + radius + 1
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                continue
            patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
            if patch.shape != (size, size) or not np.all(np.isfinite(patch)):
                continue
            border = np.concatenate((patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]))
            signal = np.maximum(patch - np.median(border), 0.0)
            total = float(np.sum(signal))
            if not math.isfinite(total) or total <= 0:
                continue
            yy, xx = np.indices(signal.shape, dtype=np.float64)
            centroid_x = x0 + float(np.sum(signal * xx) / total)
            centroid_y = y0 + float(np.sum(signal * yy) / total)
            if math.hypot(centroid_x - x, centroid_y - y) > radius * 0.5:
                continue
            refined.append((centroid_x, centroid_y))
            retained.append(index)
        return (
            np.asarray(refined, dtype=np.float64).reshape((-1, 2)),
            np.asarray(retained, dtype=np.int64),
        )

    def test_matches_per_star_loop(self) -> None:
        rng = np.random.default_rng(17)
        height, width = 160, 200
        image = rng.normal(100.0, 3.0, (height, width)).astype(np.float32)
        yy, xx = np.indices((height, width), dtype=np.float64)
        truth = rng.uniform(-4.0, 4.0 + 0, (60, 2)) + np.column_stack(
            (rng.uniform(0, width, 60), rng.uniform(0, height, 60))
        )
        for x, y in truth:
            image += (rng.uniform(200, 4000) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4.5)).astype(np.float32)
        image[40:45, 60:66] = np.nan
        image[100:112, 150:162] = 50.0  # flat, background-only patch
        approximate = truth + rng.normal(0.0, 0.4, truth.shape)
        approximate = np.vstack((approximate, [[62.0, 42.0], [156.0, 106.0], [-3.0, 10.0], [width - 1.0, 5.0], [30.5, 12.5]]))
        approximate[7] += 6.0  # displaced beyond half the radius
        for radius in (3, 5):
            with self.subTest(radius=radius):
                expected_points, expected_indices = self._reference(image, approximate, radius)
                points, indices = registration_pipeline._local_centroids(image, approximate, radius)
                np.testing.assert_array_equal(indices, expected_indices)
                np.testing.assert_array_equal(points, expected_points)
                self.assertGreater(indices.size, 30)
                self.assertLess(indices.size, approximate.shape[0])

    def test_core_window_follows_the_star_core_not_its_wings(self) -> None:
        """An asymmetric PSF must not pull the registration centroid off the core."""

        height, width = 64, 64
        yy, xx = np.indices((height, width), dtype=np.float64)
        core_x, core_y = 31.3, 30.6
        core = 4000.0 * np.exp(-((xx - core_x) ** 2 + (yy - core_y) ** 2) / (2 * 1.6**2))
        # A coma-like wing: a broad, fainter lobe displaced to the left.
        wing = 200.0 * np.exp(-((xx - (core_x - 5.0)) ** 2 + (yy - core_y) ** 2) / (2 * 4.0**2))
        symmetric = (100.0 + core).astype(np.float32)
        asymmetric = (100.0 + core + wing).astype(np.float32)
        approximate = np.asarray([[31.0, 31.0]], dtype=np.float64)

        plain_symmetric, _ = registration_pipeline._local_centroids(symmetric, approximate, 7)
        windowed_symmetric, _ = registration_pipeline._local_centroids(
            symmetric, approximate, 7, window_sigma=2.0
        )
        np.testing.assert_allclose(plain_symmetric, [[core_x, core_y]], atol=2e-3)
        np.testing.assert_allclose(windowed_symmetric, [[core_x, core_y]], atol=2e-3)

        plain_asymmetric, _ = registration_pipeline._local_centroids(asymmetric, approximate, 7)
        windowed_asymmetric, _ = registration_pipeline._local_centroids(
            asymmetric, approximate, 7, window_sigma=2.0
        )
        self.assertLess(plain_asymmetric[0, 0], core_x - 0.5)  # pulled into the wing
        self.assertAlmostEqual(windowed_asymmetric[0, 0], core_x, delta=0.12)
        self.assertAlmostEqual(windowed_asymmetric[0, 1], core_y, delta=0.02)
        self.assertLess(
            abs(windowed_asymmetric[0, 0] - core_x), 0.25 * abs(plain_asymmetric[0, 0] - core_x)
        )
        # A zero window is exactly the plain centre of mass.
        zero_window, _ = registration_pipeline._local_centroids(
            asymmetric, approximate, 7, window_sigma=0.0
        )
        np.testing.assert_array_equal(zero_window, plain_asymmetric)

    def test_full_refinement_uses_the_core_window_by_default(self) -> None:
        config = registration_pipeline.RegistrationConfig()
        self.assertEqual(
            registration_pipeline._core_window_sigma(config),
            registration_pipeline.DEFAULT_CENTROID_WINDOW_SIGMA_PX,
        )
        self.assertEqual(
            registration_pipeline._core_window_sigma(
                registration_pipeline.RegistrationConfig(full_centroid_window_sigma_px=1.5)
            ),
            1.5,
        )
        with self.assertRaises(ValueError):
            registration_pipeline.RegistrationConfig(full_centroid_window_sigma_px=-1.0)

    def test_empty_and_nonfinite_positions(self) -> None:
        image = np.ones((20, 20), dtype=np.float32)
        points, indices = registration_pipeline._local_centroids(image, np.empty((0, 2)), 3)
        self.assertEqual(points.shape, (0, 2))
        self.assertEqual(indices.size, 0)
        with self.assertRaises(ValueError):
            registration_pipeline._local_centroids(image, np.asarray([[np.nan, 5.0]]), 3)
