"""Tests for exponential backoff utility."""
import pytest

from rosclient.utils.backoff import exponential_backoff


class TestExponentialBackoff:
    """Tests for exponential_backoff function."""

    def test_basic_backoff(self):
        """Test basic exponential backoff calculation."""
        base = 1.0
        max_backoff = 30.0
        
        # First attempt
        result = exponential_backoff(base, 1, max_backoff)
        assert 0 <= result <= base * 1.2  # base + jitter
        
        # Second attempt
        result = exponential_backoff(base, 2, max_backoff)
        assert 0 <= result <= (base * 2) * 1.2  # base * 2 + jitter
        
        # Third attempt
        result = exponential_backoff(base, 3, max_backoff)
        assert 0 <= result <= (base * 4) * 1.2  # base * 4 + jitter

    def test_max_backoff_limit(self):
        """Test that backoff respects max_backoff limit."""
        base = 10.0
        max_backoff = 20.0
        
        # Attempt that would exceed max_backoff
        result = exponential_backoff(base, 5, max_backoff)
        assert result <= max_backoff * 1.2  # max + jitter

    def test_jitter_included(self):
        """Test that jitter is included in result."""
        base = 1.0
        max_backoff = 30.0
        
        results = []
        for _ in range(10):
            result = exponential_backoff(base, 1, max_backoff)
            results.append(result)
        
        # Results should vary due to jitter
        assert len(set(results)) > 1  # At least some variation

    def test_custom_jitter_fraction(self):
        """Test custom jitter fraction."""
        base = 1.0
        max_backoff = 30.0
        jitter_fraction = 0.5
        
        result = exponential_backoff(base, 1, max_backoff, jitter_fraction)
        assert 0 <= result <= base * (1 + jitter_fraction)

    def test_zero_jitter(self):
        """Test with zero jitter."""
        base = 1.0
        max_backoff = 30.0
        jitter_fraction = 0.0
        
        result = exponential_backoff(base, 1, max_backoff, jitter_fraction)
        assert result == base  # No jitter, exact base value

    def test_exponential_growth(self):
        """Test that backoff grows exponentially."""
        base = 1.0
        max_backoff = 100.0
        
        results = []
        for attempt in range(1, 6):
            result = exponential_backoff(base, attempt, max_backoff, jitter_fraction=0.0)
            results.append(result)
        
        # Each result should be approximately double the previous
        for i in range(1, len(results)):
            assert results[i] >= results[i-1] * 1.5  # Allow some tolerance

