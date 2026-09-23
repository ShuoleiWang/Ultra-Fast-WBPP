# Scientific acceptance tools

`evaluate_masters.py` measures native-grid master differences against a reference.
The validity gate and every measured failure affect the summary; candidate
cross-channel failures affect the run verdict. Missing faint/LSB flux evidence
cannot establish BETTER. The current measurement implementation still lacks the
common-footprint ratio, own/common STF agreement and LSB gate: reports list these
as `unmeasured_standard_gates` and cannot certify EQUIVALENT/BETTER. Historical
positive summary labels need reevaluation with complete evidence.

`master_tolerance_gate.py` compares baseline/candidate masters and per-metric
status changes. It does not establish equivalence to a reference pipeline by
itself. Old `benchmarks/` command paths remain compatibility launchers.

Reports and private images stay outside Git. Regression tests live in the
repository `tests/test_master_evaluation.py` and `test_master_tolerance_gate.py`.

`blink_display_study.py` creates a read-only, create-only HTML comparison of the existing Blink display and the `blink-complementary-display-v2` algorithm. It requires a content-bound Blink manifest and explicit matching calibration masters. It renders all four channels, records relative signal/noise and background differences, and can read corresponding original-pixel crops from mono FITS files. Human labels are optional, hidden by default, and never used by the algorithm. The desktop uses the same implementation for v2 previews; see [the design and acceptance boundary](../../docs/blink-display-redesign.md). Generated galleries and reports stay outside Git.
