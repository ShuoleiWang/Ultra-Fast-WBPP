"""Compatibility import for :mod:`openastroflow_engine.workflows.single_target`."""
import sys
from .workflows import single_target as _implementation
sys.modules[__name__] = _implementation
