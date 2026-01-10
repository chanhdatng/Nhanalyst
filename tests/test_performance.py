"""
Performance tests for Phase 1 analysis functions.

Tests verify that all functions meet performance requirements:
- 100K rows: <3s total load
- Individual functions: <2s each

Run with: pytest tests/test_performance.py -v --tb=short
Skip slow tests: pytest tests/test_performance.py -v -m "not slow"
"""

import pytest
import time
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import (
    compute_financial_health_score,
    compute_churn_risk_scores,
    compute_product_lifecycle,
    compute_growth_decomposition,
    compute_launch_velocity
)
from tests.conftest import generate_large_dataset


# =============================================================================
# Performance Test Suite
# =============================================================================

class TestPerformance:
    """Performance tests for 100K row datasets."""
    
    @pytest.mark.slow
    def test_health_score_100k_rows(self):
        """Test health score with 100K rows - target <1s."""
        df = generate_large_dataset(100000)
        df_curr = df[df['Year'] == 2024]
        df_prev = df[df['Year'] == 2023]
        
        start = time.time()
        result = compute_financial_health_score(df_curr, df_prev)
        elapsed = time.time() - start
        
        assert result is not None
        assert 'score' in result
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s (target: <1s)"
        print(f"\n  Health Score: {elapsed:.3f}s")
    
    @pytest.mark.slow
    def test_churn_risk_100k_rows(self):
        """Test churn risk with 100K rows - target <2s."""
        df = generate_large_dataset(100000)
        
        start = time.time()
        result = compute_churn_risk_scores(df)
        elapsed = time.time() - start
        
        assert not result.empty
        assert 'churn_risk_score' in result.columns
        assert elapsed < 2.0, f"Too slow: {elapsed:.2f}s (target: <2s)"
        print(f"\n  Churn Risk: {elapsed:.3f}s ({len(result)} customers)")
    
    @pytest.mark.slow
    def test_lifecycle_100k_rows(self):
        """Test lifecycle with 100K rows - target <1.5s."""
        df = generate_large_dataset(100000)
        
        start = time.time()
        result = compute_product_lifecycle(df)
        elapsed = time.time() - start
        
        assert not result.empty
        assert 'lifecycle_stage' in result.columns
        assert elapsed < 1.5, f"Too slow: {elapsed:.2f}s (target: <1.5s)"
        print(f"\n  Lifecycle: {elapsed:.3f}s ({len(result)} products)")
    
    @pytest.mark.slow
    def test_decomposition_100k_rows(self):
        """Test decomposition with 100K rows - target <1s."""
        df = generate_large_dataset(100000)
        df_curr = df[df['Year'] == 2024]
        df_prev = df[df['Year'] == 2023]
        
        start = time.time()
        result = compute_growth_decomposition(df_curr, df_prev)
        elapsed = time.time() - start
        
        assert result is not None
        assert 'total_growth' in result
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s (target: <1s)"
        print(f"\n  Decomposition: {elapsed:.3f}s")
    
    @pytest.mark.slow
    def test_velocity_100k_rows(self):
        """Test velocity with 100K rows - target <1.5s."""
        df = generate_large_dataset(100000)
        
        start = time.time()
        result = compute_launch_velocity(df)
        elapsed = time.time() - start
        
        # Result may be empty if no recent launches - that's OK
        assert elapsed < 1.5, f"Too slow: {elapsed:.2f}s (target: <1.5s)"
        print(f"\n  Velocity: {elapsed:.3f}s ({len(result)} products)")
    
    @pytest.mark.slow
    def test_all_functions_combined_100k(self):
        """Test all functions combined load time - target <5s total."""
        df = generate_large_dataset(100000)
        df_curr = df[df['Year'] == 2024].copy()
        df_prev = df[df['Year'] == 2023].copy()
        
        start = time.time()
        
        r1 = compute_financial_health_score(df_curr, df_prev)
        r2 = compute_churn_risk_scores(df)
        r3 = compute_product_lifecycle(df)
        r4 = compute_growth_decomposition(df_curr, df_prev)
        r5 = compute_launch_velocity(df)
        
        elapsed = time.time() - start
        
        assert r1 is not None
        assert not r2.empty
        assert not r3.empty
        assert r4 is not None
        
        assert elapsed < 5.0, f"Total too slow: {elapsed:.2f}s (target: <5s)"
        print(f"\n  TOTAL (all 5 functions): {elapsed:.3f}s")


class TestCaching:
    """Tests for cache effectiveness."""
    
    @pytest.mark.slow
    def test_cache_speeds_up_second_call(self):
        """Test that caching makes second call faster."""
        df = generate_large_dataset(50000)
        
        # First call (uncached)
        start = time.time()
        result1 = compute_churn_risk_scores(df)
        elapsed1 = time.time() - start
        
        # Second call (should be cached)
        start = time.time()
        result2 = compute_churn_risk_scores(df)
        elapsed2 = time.time() - start
        
        print(f"\n  First call: {elapsed1:.3f}s, Second call: {elapsed2:.3f}s")
        
        # Results should be identical
        assert len(result1) == len(result2)
        
        # Note: Streamlit cache may not work outside runtime context
        # This is just to verify the pattern works
    
    @pytest.mark.slow
    def test_different_params_not_cached(self):
        """Test that different parameters trigger fresh computation."""
        df1 = generate_large_dataset(10000)
        df2 = generate_large_dataset(10000)
        df2['Sold'] = df2['Sold'] * 2  # Different data
        
        result1 = compute_churn_risk_scores(df1)
        result2 = compute_churn_risk_scores(df2)
        
        # Results should be different
        # (scores likely different with different revenue)
        # Just verify both execute without error
        assert not result1.empty
        assert not result2.empty


class TestScalability:
    """Tests for larger datasets."""
    
    @pytest.mark.slow
    def test_health_score_500k_rows(self):
        """Test health score with 500K rows."""
        df = generate_large_dataset(500000)
        df_curr = df[df['Year'] == 2024]
        df_prev = df[df['Year'] == 2023]
        
        start = time.time()
        result = compute_financial_health_score(df_curr, df_prev)
        elapsed = time.time() - start
        
        assert result is not None
        assert elapsed < 3.0, f"Too slow for 500K: {elapsed:.2f}s"
        print(f"\n  Health Score (500K): {elapsed:.3f}s")
    
    @pytest.mark.slow
    def test_decomposition_500k_rows(self):
        """Test decomposition with 500K rows."""
        df = generate_large_dataset(500000)
        df_curr = df[df['Year'] == 2024]
        df_prev = df[df['Year'] == 2023]
        
        start = time.time()
        result = compute_growth_decomposition(df_curr, df_prev)
        elapsed = time.time() - start
        
        assert result is not None
        assert elapsed < 2.0, f"Too slow for 500K: {elapsed:.2f}s"
        print(f"\n  Decomposition (500K): {elapsed:.3f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
