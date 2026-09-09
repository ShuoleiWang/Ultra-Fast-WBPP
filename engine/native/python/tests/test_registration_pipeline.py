from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from skimage.transform import AffineTransform as SkimageAffineTransform

import openastroflow_registration.pipeline as registration_pipeline
from openastroflow_registration.pipeline import (
    CalibrationPlan,
    calibrate_image,
    common_autocrop,
    RegistrationConfig,
    warp_image,
)
from openastroflow_registration import (
    estimate_stellar_scale_hints,
    normalize_quality_weights,
)
from openastroflow_registration.pipeline import FrameAnalysis, StarCatalog


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

    def test_identity_autocrop_respects_interpolation_margin(self) -> None:
        crop = common_autocrop(
            [np.eye(3), np.eye(3)],
            [(100, 200), (100, 200)],
            (100, 200),
            margin=2,
        )
        self.assertEqual(crop, (2, 2, 198, 98))

    def test_translation_common_footprint(self) -> None:
        # Source 2 shifts +10 x and +5 y into reference coordinates.
        translated = np.asarray([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=float)
        crop = common_autocrop(
            [np.eye(3), translated],
            [(100, 200), (100, 200)],
            (100, 200),
            margin=0,
        )
        self.assertEqual(crop, (10, 5, 200, 100))

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
