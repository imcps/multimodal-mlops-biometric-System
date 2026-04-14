import time
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)


@contextmanager
def log_timing(name: str):
    """
    Context manager to log execution time of a code block.

    Example:
        with log_timing("forward_backward"):
            output = model(batch)
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    logging.info(f"[TIMER] {name}: {elapsed:.4f}s")