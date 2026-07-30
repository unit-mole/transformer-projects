from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timer(result: dict, key: str):
    start = perf_counter()
    try:
        yield
    finally:
        result[key] = round((perf_counter() - start) * 1000, 3)
