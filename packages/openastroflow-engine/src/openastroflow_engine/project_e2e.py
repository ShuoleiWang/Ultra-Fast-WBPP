"""Compatibility import for :mod:`openastroflow_engine.workflows.project`."""
import sys
from .workflows import project as _implementation
sys.modules[__name__] = _implementation
