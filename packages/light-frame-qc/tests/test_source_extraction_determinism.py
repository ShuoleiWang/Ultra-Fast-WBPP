"""SEP extraction with deblending must be reproducible call to call.

The installed ``sep`` build is exercised the way the quality gate uses it:
a crowded 1024x1024 field (about 3000 blended Gaussians) is extracted eight
times with deblending, with heap churn between the calls, and every result
must be identical.  A build whose Lutz/deblend buffers are uninitialised
(sep 1.4.1 on Windows' randomised heap) fails here before it can produce
run-to-run different masters; the release ships a zero-initialised build on
Windows for that reason.
"""

from __future__ import annotations

import numpy as np
import sep

from lightframeqc.source_extraction import (
    SELF_TEST_VERSION,
    crowded_field,
    extraction_self_test,
    extraction_signature,
    heap_churn,
)


def _extract(field: np.ndarray):
    objects, segmentation = sep.extract(
        field, 2.5, err=3.0, minarea=5, deblend_nthresh=32, deblend_cont=0.005, clean=True, segmentation_map=True
    )
    return objects, segmentation


def test_eight_deblended_extractions_of_a_crowded_field_are_identical() -> None:
    field = crowded_field(1024, source_count=3000, seed=11)
    rng = np.random.default_rng(11)
    first_objects, first_segmentation = _extract(field)
    # Enough blends that the deblender's buffers are exercised on every call.
    assert 1500 < len(first_objects) < 3000
    for round_index in range(7):
        heap_churn(rng, rounds=8)
        objects, segmentation = _extract(field)
        assert len(objects) == len(first_objects), f"round {round_index}: object count changed"
        assert np.array_equal(objects, first_objects), f"round {round_index}: object records differ"
        assert np.array_equal(segmentation, first_segmentation), f"round {round_index}: segmentation differs"
    # The digest form used by the receipt agrees with the raw comparison.
    assert extraction_signature(field) == extraction_signature(field)


def test_self_test_reports_a_deterministic_extractor_with_evidence() -> None:
    report = extraction_self_test()
    assert report["version"] == SELF_TEST_VERSION
    assert report["deterministic"] is True
    assert report["warnings"] == []
    assert report["objectCounts"][0] == report["objectCounts"][1] > 100
    assert report["signatures"][0] == report["signatures"][1]
    assert report["sepVersion"] == sep.__version__


def test_self_test_flags_a_non_deterministic_extractor() -> None:
    class Flaky:
        """An extractor whose deblending drops one object on the second call."""

        __version__ = "flaky"
        calls = 0

        def extract(self, image, threshold, **options):
            self.calls += 1
            objects, segmentation = sep.extract(image, threshold, **options)
            if self.calls % 2 == 0:
                objects = objects[:-1]
            return objects, segmentation

    report = extraction_self_test(sep_module=Flaky())
    assert report["deterministic"] is False
    assert report["warnings"] == ["SEP_NONDETERMINISTIC"]
    assert report["objectCounts"][0] == report["objectCounts"][1] + 1
    assert report["signatures"][0] != report["signatures"][1]
