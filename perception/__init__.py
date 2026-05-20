"""Compatibility namespace for legacy ``perception.*`` imports.

The refactor split perception code across ``branch1`` and ``branch2``.  This
package preserves the old import names while the entry points are cleaned up.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PERCEPTION_DIRS = [
    _ROOT / "branch1" / "processing",
    _ROOT / "branch1" / "calib",
    _ROOT / "branch1" / "inputs" / "geometry",
    _ROOT / "branch1" / "inputs" / "recorders",
    _ROOT / "branch2",
]

for _path in _PERCEPTION_DIRS:
    if _path.exists():
        _str = str(_path)
        if _str not in __path__:
            __path__.append(_str)
        if _str not in sys.path:
            sys.path.insert(0, _str)

