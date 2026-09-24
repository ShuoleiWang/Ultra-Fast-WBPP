"""The run tracer: nested timers with exclusive time, stage records from
progress events, worker event collection and the summary table."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

REPOSITORY = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("trace_run", REPOSITORY / "benchmarks" / "trace_run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_timers_record_nested_calls_with_exclusive_time() -> None:
    trace_run = _load_module()
    tracer = trace_run.Tracer(time.perf_counter_ns())

    def inner() -> int:
        time.sleep(0.02)
        return 1

    def outer() -> int:
        time.sleep(0.01)
        return inner() + inner()

    traced_inner = tracer.wrap(inner, "inner", "test")
    traced_outer = tracer.wrap(lambda: traced_inner() + traced_inner() + (time.sleep(0.01) or 0), "outer", "test")
    assert traced_outer() == 2
    summary = trace_run.summarize(tracer.events)
    assert summary["inner"]["calls"] == 2
    assert summary["outer"]["calls"] == 1
    assert summary["outer"]["inclusiveSeconds"] >= summary["inner"]["inclusiveSeconds"]
    # The outer timer's exclusive time excludes its two traced children.
    assert summary["outer"]["exclusiveSeconds"] < summary["outer"]["inclusiveSeconds"]
    assert 0.005 < summary["outer"]["exclusiveSeconds"] < 0.2
    for event in tracer.events:
        assert event["ph"] == "X" and event["tid"] == threading.get_ident()
    del outer


def test_stage_records_come_from_progress_events_and_worker_files_merge(tmp_path: Path) -> None:
    trace_run = _load_module()
    tracer = trace_run.Tracer(time.perf_counter_ns())
    tracer.record_stage({"stage": "registration", "status": "started", "message": "measuring", "current": 0, "total": 0})
    time.sleep(0.01)
    tracer.record_stage({"stage": "registration", "status": "completed", "message": "accepted 3", "current": 3, "total": 3})
    # A project stage that only reports "running" then "completed".
    tracer.record_stage({"stage": "color", "status": "running", "message": "rendering", "current": 0, "total": 1})
    tracer.record_stage({"stage": "color", "status": "completed", "message": "done", "current": 1, "total": 1})
    assert [record["stage"] for record in tracer.stage_records] == ["registration", "color"]
    assert tracer.stage_records[0]["wallSeconds"] >= 0.01
    assert tracer.stage_records[0]["message"] == "accepted 3"
    stage_events = [event for event in tracer.events if event.get("cat") == "stage"]
    assert [event["name"] for event in stage_events] == ["registration", "color"]

    worker_dir = tmp_path / "trace.workers"
    worker_dir.mkdir()
    (worker_dir / "worker-123.json").write_text(
        json.dumps({"pid": 123, "events": [{"name": "qc.measure_frame", "cat": "qc", "ph": "X", "ts": 1.0, "dur": 500.0, "pid": 123, "tid": 5, "args": {"exclusive_us": 500.0}}]}),
        encoding="utf-8",
    )
    events = trace_run.collect_worker_events(worker_dir)
    assert any(event["name"] == "qc.measure_frame" and event["args"]["worker"] for event in events)
    assert any(event["ph"] == "M" and event["pid"] == 123 for event in events)
    tracer.events.extend(events)
    summary = trace_run.summarize(tracer.events)
    assert summary["qc.measure_frame"]["workerProcesses"] == 1
    table = trace_run.format_table(tracer.stage_records, summary, total_wall=1.5)
    assert "registration" in table and "qc.measure_frame" in table and "total wall 1.50 s" in table

    trace_path = tmp_path / "trace.json"
    tracer.dump(trace_path, {"command": ["run-project"]})
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["command"] == ["run-project"]
    assert payload["traceEvents"][0]["ph"] == "M"


def test_install_reports_missing_targets_and_wraps_once() -> None:
    trace_run = _load_module()
    tracer = trace_run.Tracer(time.perf_counter_ns())
    tracer.install((("json", "dumps", "json.dumps", "test"), ("json", "no_such_function", "json.missing", "test")))
    assert "json.dumps" in tracer.installed
    assert tracer.missing == ["json.no_such_function"]
    try:
        assert json.dumps({"a": 1}) == '{"a": 1}'
        assert getattr(json.dumps, "__ufwbpp_traced__", False) is True
        tracer.install((("json", "dumps", "json.dumps", "test"),))
        assert tracer.installed.count("json.dumps") == 1
        assert trace_run.summarize(tracer.events)["json.dumps"]["calls"] == 1
    finally:
        json.dumps = json.dumps.__wrapped__  # type: ignore[attr-defined]
    assert not hasattr(json.dumps, "__ufwbpp_traced__")
    assert sys.modules["json"].dumps is json.dumps


def test_every_timer_target_still_exists() -> None:
    """A renamed or moved function must not silently drop out of the trace."""

    trace_run = _load_module()
    missing = []
    for module_name, attribute, *_rest in trace_run.TARGETS:
        module = importlib.import_module(module_name)
        if not hasattr(module, attribute.split(".")[0]):
            missing.append(f"{module_name}.{attribute}")
    assert missing == []
