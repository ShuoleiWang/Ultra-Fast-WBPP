"""Compatibility command; implementation: tools/validation/master_tolerance_gate.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validation.master_tolerance_gate import main

if __name__ == "__main__":
    raise SystemExit(main())
