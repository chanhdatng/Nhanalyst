"""
Edge case tests for Phase 1 analysis functions.

Tests validate graceful handling of:
- Empty DataFrames
- Null/missing values
- Single entity scenarios
- Extreme values (very high/low)
- Division by zero

Run with: pytest tests/test_edge_cases.py -v
"""

import pytest
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


# =============================================================================
# Edge Case Test Suite
# =============================================================================

class TestEmptyDataFrames:
    """Tests for empty DataFrame handling."""
    
    def test_health_score_both_empty(self):
        """Test health score with both DataFrames empty."""
        empty = pd.DataFrame()
        result = compute_financial_health_score(empty, empty)
        
        assert result is not None
        assert 'score' in result
        assert result['score'] >= 0
    
    def test_health_score_curr_empty(self):
        """Test health score with current period empty."""
        empty = pd.DataFrame()
        df_prev = pd.DataFrame({
            'date__ym': pd.date_range('2023-01-01', periods=5, freq='ME'),
            'Sold': [1000] * 5,
            'Quantity (KG)': [100] * 5,
            'Name of client': ['Client A'] * 5,
            'Name of product': ['Product X'] * 5,
            'Year': [2023] * 5,
            'Month': list(range(1, 6))
        })
        
        result = compute_financial_health_score(empty, df_prev)
        assert result is not None
    
    def test_health_score_prev_empty(self):
        """Test health score with previous period empty (common case)."""
        df_curr = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=5, freq='ME'),
            'Sold': [1000] * 5,
            'Quantity (KG)': [100] * 5,
            'Name of client': ['Client A'] * 5,
            'Name of product': ['Product X'] * 5,
            'Year': [2024] * 5,
            'Month': list(range(1, 6))
        })
        
        result = compute_financial_health_score(df_curr, pd.DataFrame())
        assert result is not None
        assert result['score'] >= 0
    
    def test_churn_risk_empty(self):
        """Test churn risk with empty DataFrame."""
        result = compute_churn_risk_scores(pd.DataFrame())
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty or len(result) == 0
    
    def test_lifecycle_empty(self):
        """Test lifecycle with empty DataFrame."""
        result = compute_product_lifecycle(pd.DataFrame())
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_decomposition_empty(self):
        """Test decomposition with empty DataFrames."""
        result = compute_growth_decomposition(pd.DataFrame(), pd.DataFrame())
        
        # Should return None when no data
        assert result is None
    
    def test_velocity_empty(self):
        """Test velocity with empty DataFrame."""
        result = compute_launch_velocity(pd.DataFrame())
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestSingleEntity:
    """Tests for single customer/product scenarios."""
    
    def test_churn_single_customer(self):
        """Test churn risk with single customer."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=12, freq='ME'),
            'Sold': [1000] * 12,
            'Quantity (KG)': [100] * 12,
            'Name of client': ['Single Client'] * 12,
            'Name of product': ['Product X'] * 12,
            'Year': [2024] * 12,
            'Month': list(range(1, 13))
        })
        
        result = compute_churn_risk_scores(df)
        
        assert len(result) == 1
        assert result['Name of client'].iloc[0] == 'Single Client'
        assert 0 <= result['churn_risk_score'].iloc[0] <= 100
    
    def test_lifecycle_single_product(self):
        """Test lifecycle with single product."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=6, freq='ME'),
            'Sold': [1000, 2000, 3000, 4000, 5000, 6000],
            'Quantity (KG)': [100] * 6,
            'Name of client': ['Client A'] * 6,
            'Name of product': ['Single Product'] * 6,
            'Year': [2024] * 6,
            'Month': list(range(1, 7))
        })
        
        result = compute_product_lifecycle(df)
        
        assert len(result) == 1
        assert result['Name of product'].iloc[0] == 'Single Product'
        assert result['lifecycle_stage'].iloc[0] in ['Introduction', 'Growth', 'Maturity', 'Decline']
    
    def test_decomposition_single_customer_each(self):
        """Test decomposition with single customer in each period."""
        df_prev = pd.DataFrame({
            'date__ym': pd.date_range('2023-01-01', periods=12, freq='ME'),
            'Sold': [1000.0] * 12,
            'Quantity (KG)': [100.0] * 12,
            'Name of client': ['Old Client'] * 12,
            'Name of product': ['Product A'] * 12,
            'Year': [2023] * 12,
            'Month': list(range(1, 13))
        })
        
        df_curr = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=12, freq='ME'),
            'Sold': [2000.0] * 12,
            'Quantity (KG)': [200.0] * 12,
            'Name of client': ['New Client'] * 12,
            'Name of product': ['Product A'] * 12,
            'Year': [2024] * 12,
            'Month': list(range(1, 13))
        })
        
        result = compute_growth_decomposition(df_curr, df_prev)
        
        assert result is not None
        assert 'components' in result
        # Should show new customers and churn
        assert result['components']['new_customers'] > 0
        assert result['components']['churn'] < 0


class TestNullValues:
    """Tests for null/missing value handling."""
    
    def test_health_score_null_revenue(self):
        """Test health score with null revenue values."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=10, freq='ME'),
            'Sold': [None, 1000, None, 2000, None, 3000, None, 4000, None, 5000],
            'Quantity (KG)': [100] * 10,
            'Name of client': ['Client A'] * 10,
            'Name of product': ['Product X'] * 10,
            'Year': [2024] * 10,
            'Month': list(range(1, 11))
        })
        
        result = compute_financial_health_score(df, pd.DataFrame())
        
        assert result is not None
        assert not pd.isna(result['score'])
    
    def test_churn_risk_null_values(self):
        """Test churn risk with null values in various columns."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=10, freq='ME'),
            'Sold': [1000, None, 2000, None, 3000, 4000, 5000, 6000, 7000, 8000],
            'Quantity (KG)': [100, None, 100, 100, None, 100, 100, 100, 100, 100],
            'Name of client': ['Client A'] * 5 + ['Client B'] * 5,
            'Name of product': ['Product X', None, 'Product X', 'Product Y', None] * 2,
            'Year': [2024] * 10,
            'Month': list(range(1, 11))
        })
        
        result = compute_churn_risk_scores(df)
        
        assert not result.empty
        assert result['churn_risk_score'].notna().all()
    
    def test_lifecycle_null_dates(self):
        """Test lifecycle with null dates."""
        df = pd.DataFrame({
            'date__ym': [pd.Timestamp('2024-01-01'), None, pd.Timestamp('2024-03-01'),
                        pd.Timestamp('2024-04-01'), None, pd.Timestamp('2024-06-01')],
            'Sold': [1000, 2000, 3000, 4000, 5000, 6000],
            'Quantity (KG)': [100] * 6,
            'Name of client': ['Client A'] * 6,
            'Name of product': ['Product X'] * 6,
            'Year': [2024] * 6,
            'Month': [1, 2, 3, 4, 5, 6]
        })
        
        # Should handle gracefully (may return empty or skip nulls)
        result = compute_product_lifecycle(df)
        assert isinstance(result, pd.DataFrame)


class TestExtremeValues:
    """Tests for extreme value handling."""
    
    def test_health_score_very_high_revenue(self):
        """Test health score with very high revenue values."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=5, freq='ME'),
            'Sold': [1e12, 2e12, 3e12, 4e12, 5e12],  # Trillions
            'Quantity (KG)': [1e9] * 5,
            'Name of client': ['Client A'] * 5,
            'Name of product': ['Product X'] * 5,
            'Year': [2024] * 5,
            'Month': list(range(1, 6))
        })
        
        result = compute_financial_health_score(df, pd.DataFrame())
        
        assert result is not None
        assert 0 <= result['score'] <= 100
        assert not np.isinf(result['score'])
    
    def test_health_score_very_low_revenue(self):
        """Test health score with very small revenue values."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=5, freq='ME'),
            'Sold': [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],  # Very small
            'Quantity (KG)': [1e-8] * 5,
            'Name of client': ['Client A'] * 5,
            'Name of product': ['Product X'] * 5,
            'Year': [2024] * 5,
            'Month': list(range(1, 6))
        })
        
        result = compute_financial_health_score(df, pd.DataFrame())
        
        assert result is not None
        assert 0 <= result['score'] <= 100
    
    def test_churn_risk_negative_values(self):
        """Test churn risk with negative revenue (returns/refunds)."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=12, freq='ME'),
            'Sold': [1000, -500, 2000, -200, 1500, 1800, 2100, -100, 2500, 2800, 3000, 3200],
            'Quantity (KG)': [100] * 12,
            'Name of client': ['Client A'] * 12,
            'Name of product': ['Product X'] * 12,
            'Year': [2024] * 12,
            'Month': list(range(1, 13))
        })
        
        result = compute_churn_risk_scores(df)
        
        assert not result.empty
        assert result['churn_risk_score'].between(0, 100).all()
    
    def test_lifecycle_zero_revenue(self):
        """Test lifecycle with zero revenue months."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=12, freq='ME'),
            'Sold': [0, 0, 0, 1000, 2000, 3000, 0, 0, 5000, 6000, 7000, 8000],
            'Quantity (KG)': [100] * 12,
            'Name of client': ['Client A'] * 12,
            'Name of product': ['Product X'] * 12,
            'Year': [2024] * 12,
            'Month': list(range(1, 13))
        })
        
        result = compute_product_lifecycle(df)
        
        assert not result.empty
        assert result['lifecycle_stage'].iloc[0] in ['Introduction', 'Growth', 'Maturity', 'Decline']


class TestDivisionByZero:
    """Tests for division by zero scenarios."""
    
    def test_health_score_zero_prev_revenue(self):
        """Test health score when previous revenue is zero."""
        df_curr = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=5, freq='ME'),
            'Sold': [1000, 2000, 3000, 4000, 5000],
            'Quantity (KG)': [100] * 5,
            'Name of client': ['Client A'] * 5,
            'Name of product': ['Product X'] * 5,
            'Year': [2024] * 5,
            'Month': list(range(1, 6))
        })
        
        df_prev = pd.DataFrame({
            'date__ym': pd.date_range('2023-01-01', periods=5, freq='ME'),
            'Sold': [0, 0, 0, 0, 0],  # Zero revenue
            'Quantity (KG)': [0] * 5,
            'Name of client': ['Client A'] * 5,
            'Name of product': ['Product X'] * 5,
            'Year': [2023] * 5,
            'Month': list(range(1, 6))
        })
        
        result = compute_financial_health_score(df_curr, df_prev)
        
        assert result is not None
        assert not np.isnan(result['score'])
        assert not np.isinf(result['score'])
    
    def test_decomposition_zero_prev_revenue(self):
        """Test decomposition when previous revenue is zero."""
        df_curr = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=1, freq='ME'),
            'Sold': [10000.0],
            'Quantity (KG)': [1000.0],
            'Name of client': ['Client A'],
            'Name of product': ['Product X'],
            'Year': [2024],
            'Month': [1]
        })
        
        df_prev = pd.DataFrame({
            'date__ym': pd.date_range('2023-01-01', periods=1, freq='ME'),
            'Sold': [0.0],
            'Quantity (KG)': [0.0],
            'Name of client': ['Client A'],
            'Name of product': ['Product X'],
            'Year': [2023],
            'Month': [1]
        })
        
        result = compute_growth_decomposition(df_curr, df_prev)
        
        # May return None if prev is empty/zero, or handle gracefully
        if result is not None:
            assert not np.isnan(result['total_growth'])
            assert not np.isinf(result['total_growth'])
    
    def test_velocity_zero_m1_revenue(self):
        """Test velocity when M1 revenue is zero."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-09-01', periods=4, freq='ME'),
            'Sold': [0, 1000, 2000, 3000],  # Zero M1
            'Quantity (KG)': [0, 100, 200, 300],
            'Name of client': ['Client A'] * 4,
            'Name of product': ['New Product'] * 4,
            'Year': [2024] * 4,
            'Month': [9, 10, 11, 12]
        })
        
        result = compute_launch_velocity(df)
        
        if not result.empty:
            assert result['velocity_pct'].notna().all() or result['velocity_pct'].isna().all()
            if result['velocity_pct'].notna().any():
                assert not np.isinf(result['velocity_pct'].iloc[0])


class TestSpecialCases:
    """Miscellaneous special case tests."""
    
    def test_all_same_customer(self):
        """Test when all rows are same customer."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Sold': [1000] * 100,
            'Quantity (KG)': [100] * 100,
            'Name of client': ['Same Client'] * 100,
            'Name of product': ['Product X'] * 100,
            'Year': [2024] * 100,
            'Month': [1] * 100
        })
        
        result = compute_churn_risk_scores(df)
        
        assert len(result) == 1
    
    def test_all_same_product(self):
        """Test when all rows are same product."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=12, freq='ME'),
            'Sold': [1000] * 12,
            'Quantity (KG)': [100] * 12,
            'Name of client': ['Client A'] * 12,
            'Name of product': ['Same Product'] * 12,
            'Year': [2024] * 12,
            'Month': list(range(1, 13))
        })
        
        result = compute_product_lifecycle(df)
        
        assert len(result) == 1
    
    def test_very_old_data(self):
        """Test with data from many years ago."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2010-01-01', periods=12, freq='ME'),
            'Sold': [1000] * 12,
            'Quantity (KG)': [100] * 12,
            'Name of client': ['Client A'] * 12,
            'Name of product': ['Old Product'] * 12,
            'Year': [2010] * 12,
            'Month': list(range(1, 13))
        })
        
        # Velocity uses relative dates from data, so it may return results
        # depending on implementation. Just verify no crash.
        result = compute_launch_velocity(df)
        assert isinstance(result, pd.DataFrame)
        
        # Lifecycle should still work
        lifecycle = compute_product_lifecycle(df)
        assert not lifecycle.empty
    
    def test_future_dates(self):
        """Test with future dates (forecasts)."""
        df = pd.DataFrame({
            'date__ym': pd.date_range('2030-01-01', periods=12, freq='ME'),
            'Sold': [1000] * 12,
            'Quantity (KG)': [100] * 12,
            'Name of client': ['Client A'] * 12,
            'Name of product': ['Future Product'] * 12,
            'Year': [2030] * 12,
            'Month': list(range(1, 13))
        })
        
        # Should handle gracefully
        result = compute_product_lifecycle(df)
        assert isinstance(result, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
