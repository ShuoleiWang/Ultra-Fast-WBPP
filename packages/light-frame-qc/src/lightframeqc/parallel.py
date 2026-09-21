"""Bounded per-frame parallelism shared by measurement and group analysis.

About a third of a frame's measurement and most of its analysis is
Python-level work that holds the GIL (per-cell statistics, star records,
triangle matching in astroalign), so threads reach only about 1.6x on eight
cores while spawned processes scale with the cores.  Every frame runs the
same function either way, so the choice of executor never changes a value.
"""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import os
import pickle
import sys
from typing import Any, Callable, Iterable, TypeVar
import warnings

# Below this many items a process pool's start-up outweighs its gain.
PROCESS_POOL_MINIMUM_ITEMS = 12
# ``LIGHTFRAMEQC_PARALLELISM=threads|processes`` overrides the choice.
PARALLELISM_ENVIRONMENT = "LIGHTFRAMEQC_PARALLELISM"
_PROCESS_POOL_FAILURES = (BrokenProcessPool, OSError, RuntimeError, EOFError, pickle.PicklingError)

_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


def choose_parallelism(workers: int, item_count: int) -> str:
    """``sequential``, ``threads`` or ``processes`` for a batch of items."""

    if workers <= 1 or item_count <= 1:
        return "sequential"
    requested = os.environ.get(PARALLELISM_ENVIRONMENT, "").strip().casefold()
    if requested in {"threads", "processes"}:
        return requested
    return "processes" if item_count >= PROCESS_POOL_MINIMUM_ITEMS else "threads"


class FrameRunner:
    """Run one importable function over items in order with bounded workers.

    The executor is created on first use and reused by every ``map`` until
    ``close``.  A process pool that cannot start or breaks (a frozen build
    without freeze support, an exhausted process table) is replaced by a
    thread pool, and the affected batch is computed again there; ``stats``
    records what actually ran.
    """

    def __init__(self, workers: int, item_count: int) -> None:
        if isinstance(workers, bool) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.parallelism = choose_parallelism(workers, item_count)
        self.workers = 1 if self.parallelism == "sequential" else min(workers, max(item_count, 1))
        self._executor: Executor | None = None
        self.fallback_reason: str | None = None

    @property
    def stats(self) -> dict[str, Any]:
        # ``fallbackReason`` is set when a process pool was replaced by
        # threads; receipts keep it so a slow frozen build (no spawned
        # workers) can be told apart from a machine that is merely small.
        return {
            "parallelism": self.parallelism,
            "workers": self.workers,
            "fallbackReason": self.fallback_reason,
        }

    def __enter__(self) -> "FrameRunner":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _open(self) -> Executor:
        if self._executor is None:
            if self.parallelism == "processes":
                self._executor = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=multiprocessing.get_context("spawn")
                )
            else:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.workers, thread_name_prefix="lightframeqc"
                )
        return self._executor

    def map(self, function: Callable[[_Task], _Result], tasks: Iterable[_Task]) -> list[_Result]:
        items = list(tasks)
        if self.parallelism == "sequential" or len(items) <= 1:
            return [function(item) for item in items]
        if self.parallelism == "processes":
            try:
                return list(self._open().map(function, items))
            except _PROCESS_POOL_FAILURES as error:
                self._fall_back_to_threads(error)
        return list(self._open().map(function, items))

    def _fall_back_to_threads(self, error: BaseException) -> None:
        """Replace a broken or unstartable process pool by a thread pool.

        The reason is kept for the receipt and raised as a warning because
        the fallback is silent otherwise and costs several times the wall
        time on eight cores.  A frozen executable is the expected cause: its
        spawned children must re-enter ``multiprocessing.freeze_support()``
        before any argument dispatch, or they run the command line instead.
        """

        self.close()
        self.parallelism = "threads"
        reason = f"{type(error).__name__}: {error}".strip().rstrip(":")
        if getattr(sys, "frozen", False):
            reason += (
                " (frozen executable: the launcher must call multiprocessing.freeze_support() "
                "before dispatching its arguments)"
            )
        self.fallback_reason = reason
        warnings.warn(
            f"frame worker process pool unavailable, computing in threads instead: {reason}",
            RuntimeWarning,
            stacklevel=3,
        )


__all__ = ["FrameRunner", "PARALLELISM_ENVIRONMENT", "PROCESS_POOL_MINIMUM_ITEMS", "choose_parallelism"]
