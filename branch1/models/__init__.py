"""Branch 1 model implementations and legacy flat ``models.*`` support."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODEL_DIRS = [
    _HERE,
    _HERE / "3dgcnn",
    _HERE / "hybrid_rd",
    _HERE / "kpconv",
]

for _path in _MODEL_DIRS:
    if _path.exists():
        _str = str(_path)
        if _str not in __path__:
            __path__.append(_str)
        if _str not in sys.path:
            sys.path.insert(0, _str)
