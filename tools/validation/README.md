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
