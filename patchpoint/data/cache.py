"""On-disk cache for external calls, keyed by a content hash.

GitHub, embedding, and LLM calls all go through this so an eval can be re-run
dozens of times and pay for the network/API cost once. See CLAUDE.md.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

CACHE_DIR = Path(".cache")

F = TypeVar("F", bound=Callable[..., Any])


def _cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
    return h.hexdigest()


def cached_json(namespace: str) -> Callable[[F], F]:
    """Decorator: cache a function's JSON-serializable return value on disk.

    The cache key is a hash of `namespace` plus the function's positional and
    keyword arguments (stringified), so callers should pass simple, stable types.
    """

    def decorator(fn: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key_parts = [namespace, *[str(a) for a in args]]
            key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
            key = _cache_key(*key_parts)
            path = CACHE_DIR / namespace / f"{key}.json"

            if path.exists():
                return json.loads(path.read_text())

            result = fn(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
