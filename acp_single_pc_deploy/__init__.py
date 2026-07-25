"""Import shim that keeps the package usable regardless of the clone directory name."""

from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in __path__:
    __path__.append(str(_REPOSITORY_ROOT))
