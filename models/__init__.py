"""Compatibility namespace for legacy ``models.*`` imports.

The refactor stores model files under ``branch1/models``.  Several scripts
still import the old flat ``models`` package; extending this package path keeps
those imports working without changing model behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIRS = [
    _ROOT / "branch1" / "models",
    _ROOT / "branch1" / "models" / "3dgcnn",
    _ROOT / "branch1" / "models" / "hybrid_rd",
    _ROOT / "branch1" / "models" / "kpconv",
]

for _path in _MODEL_DIRS:
    if _path.exists():
        _str = str(_path)
        if _str not in __path__:
            __path__.append(_str)
        if _str not in sys.path:
            sys.path.insert(0, _str)

