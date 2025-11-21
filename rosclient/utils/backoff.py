"""Exponential backoff utilities."""
import random


def exponential_backoff(base: float, attempt: int, max_backoff: float, jitter_fraction: float = 0.2) -> float:
    """
    Calculate exponential backoff time with jitter.
    
    Args:
        base: Base backoff time in seconds
        attempt: Current attempt number (1-indexed)
        max_backoff: Maximum backoff time in seconds
        jitter_fraction: Fraction of backoff time to use as jitter (0.0 to 1.0)
    
    Returns:
        Backoff time in seconds
    """
    backoff = min(max_backoff, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, backoff * jitter_fraction)
    return backoff + jitter

