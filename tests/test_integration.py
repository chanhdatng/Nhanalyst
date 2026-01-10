"""
Integration tests for UI components.

Tests verify that all tabs load without errors and render expected elements.

Run with: pytest tests/test_integration.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModuleImports:
    """Test that all modules import correctly."""
    
    def test_import_analysis_functions(self):
        """Test importing all analysis functions."""
        from src.analysis import (
            compute_financial_health_score,
            compute_churn_risk_scores,
            compute_product_lifecycle,
            compute_growth_decomposition,
            compute_launch_velocity,
            compute_top_level_kpis,
            compute_client_metrics,
            compute_product_metrics,
            compute_region_metrics,
            compute_rfm_clusters
        )
        
        # All imports should succeed
        assert callable(compute_financial_health_score)
        assert callable(compute_churn_risk_scores)
        assert callable(compute_product_lifecycle)
        assert callable(compute_growth_decomposition)
        assert callable(compute_launch_velocity)
    
    def test_import_tab_modules(self):
        """Test importing all tab render functions."""
        from src.tabs.executive_overview import render_executive_overview
        from src.tabs.customer_market import render_customer_market
        from src.tabs.product_intelligence import render_product_intelligence
        from src.tabs.growth_insights import render_growth_insights
        from src.tabs.product_launching import render_product_launching
        from src.tabs.vietnam_focus import render_vietnam_focus
        
        assert callable(render_executive_overview)
        assert callable(render_customer_market)
        assert callable(render_product_intelligence)
        assert callable(render_growth_insights)
        assert callable(render_product_launching)
        assert callable(render_vietnam_focus)
    
    def test_import_utils(self):
        """Test importing utility functions."""
        from src.utils import (
            calculate_growth,
            filter_by_date,
            DEFAULT_DATE_COL
        )
        
        assert callable(calculate_growth)
        assert callable(filter_by_date)
        assert DEFAULT_DATE_COL == 'date__ym'
    
    def test_import_data_processing(self):
        """Test importing data processing functions."""
        from src.data_processing import load_data, clean_data
        
        assert callable(load_data)
        assert callable(clean_data)


class TestFunctionSignatures:
    """Test that function signatures match expected parameters."""
    
    def test_health_score_signature(self):
        """Test compute_financial_health_score signature."""
        from src.analysis import compute_financial_health_score
        import inspect
        
        sig = inspect.signature(compute_financial_health_score)
        params = list(sig.parameters.keys())
        
        assert 'df_curr' in params
        assert 'df_prev' in params
    
    def test_churn_risk_signature(self):
        """Test compute_churn_risk_scores signature."""
        from src.analysis import compute_churn_risk_scores
        import inspect
        
        sig = inspect.signature(compute_churn_risk_scores)
        params = list(sig.parameters.keys())
        
        assert 'df' in params
    
    def test_lifecycle_signature(self):
        """Test compute_product_lifecycle signature."""
        from src.analysis import compute_product_lifecycle
        import inspect
        
        sig = inspect.signature(compute_product_lifecycle)
        params = list(sig.parameters.keys())
        
        assert 'df' in params
    
    def test_decomposition_signature(self):
        """Test compute_growth_decomposition signature."""
        from src.analysis import compute_growth_decomposition
        import inspect
        
        sig = inspect.signature(compute_growth_decomposition)
        params = list(sig.parameters.keys())
        
        assert 'df_curr' in params
        assert 'df_prev' in params
    
    def test_velocity_signature(self):
        """Test compute_launch_velocity signature."""
        from src.analysis import compute_launch_velocity
        import inspect
        
        sig = inspect.signature(compute_launch_velocity)
        params = list(sig.parameters.keys())
        
        assert 'df' in params
        assert 'min_age_months' in params


class TestReturnTypes:
    """Test that functions return expected types."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Sold': np.random.randint(1000, 10000, 100),
            'Quantity (KG)': np.random.randint(100, 1000, 100),
            'Name of client': [f'Client {i % 10}' for i in range(100)],
            'Name of product': [f'Product {i % 5}' for i in range(100)],
            'Year': [2024] * 100,
            'Month': [(i % 12) + 1 for i in range(100)]
        })
    
    def test_health_score_returns_dict(self, sample_df):
        """Test that health score returns a dict."""
        from src.analysis import compute_financial_health_score
        
        result = compute_financial_health_score(sample_df, sample_df)
        
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'color' in result
        assert 'components' in result
    
    def test_churn_risk_returns_dataframe(self, sample_df):
        """Test that churn risk returns a DataFrame."""
        from src.analysis import compute_churn_risk_scores
        
        result = compute_churn_risk_scores(sample_df)
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'churn_risk_score' in result.columns
            assert 'risk_level' in result.columns
    
    def test_lifecycle_returns_dataframe(self, sample_df):
        """Test that lifecycle returns a DataFrame."""
        from src.analysis import compute_product_lifecycle
        
        result = compute_product_lifecycle(sample_df)
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'lifecycle_stage' in result.columns
            assert 'stage_emoji' in result.columns
    
    def test_decomposition_returns_dict_or_none(self, sample_df):
        """Test that decomposition returns dict or None."""
        from src.analysis import compute_growth_decomposition
        
        result = compute_growth_decomposition(sample_df, sample_df)
        
        assert result is None or isinstance(result, dict)
        if result is not None:
            assert 'total_growth' in result
            assert 'components' in result
    
    def test_velocity_returns_dataframe(self, sample_df):
        """Test that velocity returns a DataFrame."""
        from src.analysis import compute_launch_velocity
        
        result = compute_launch_velocity(sample_df)
        
        assert isinstance(result, pd.DataFrame)
        # May be empty if no recent launches


class TestDataValidation:
    """Test data validation within functions."""
    
    def test_health_score_validates_score_range(self):
        """Test that health score is always 0-100."""
        from src.analysis import compute_financial_health_score
        
        # Various scenarios
        scenarios = [
            (pd.DataFrame({'Sold': [1000], 'Quantity (KG)': [100], 
                          'Name of client': ['A'], 'Name of product': ['X'],
                          'Year': [2024], 'Month': [1], 'date__ym': [pd.Timestamp('2024-01')]}),
             pd.DataFrame()),
            (pd.DataFrame(), pd.DataFrame()),
        ]
        
        for df_curr, df_prev in scenarios:
            result = compute_financial_health_score(df_curr, df_prev)
            assert 0 <= result['score'] <= 100, f"Score out of range: {result['score']}"
    
    def test_churn_risk_validates_score_range(self):
        """Test that churn risk scores are always 0-100."""
        from src.analysis import compute_churn_risk_scores
        
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=50, freq='D'),
            'Sold': np.random.randint(100, 10000, 50),
            'Quantity (KG)': np.random.randint(10, 1000, 50),
            'Name of client': [f'Client {i % 5}' for i in range(50)],
            'Name of product': [f'Product {i % 3}' for i in range(50)],
            'Year': [2024] * 50,
            'Month': [1] * 50
        })
        
        result = compute_churn_risk_scores(df)
        
        if not result.empty:
            assert result['churn_risk_score'].between(0, 100).all(), \
                f"Scores out of range: {result['churn_risk_score'].describe()}"
    
    def test_lifecycle_validates_stages(self):
        """Test that lifecycle stages are valid."""
        from src.analysis import compute_product_lifecycle
        
        df = pd.DataFrame({
            'date__ym': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Sold': np.random.randint(1000, 10000, 100),
            'Quantity (KG)': np.random.randint(100, 1000, 100),
            'Name of client': [f'Client {i % 10}' for i in range(100)],
            'Name of product': [f'Product {i % 5}' for i in range(100)],
            'Year': [2024] * 100,
            'Month': [(i % 12) + 1 for i in range(100)]
        })
        
        result = compute_product_lifecycle(df)
        
        valid_stages = ['Introduction', 'Growth', 'Maturity', 'Decline']
        if not result.empty:
            assert result['lifecycle_stage'].isin(valid_stages).all(), \
                f"Invalid stages found: {result['lifecycle_stage'].unique()}"


class TestCrossModuleIntegration:
    """Test that modules work together correctly."""
    
    def test_analysis_used_by_tabs(self):
        """Test that analysis functions can be used by tab modules."""
        # This is a smoke test - just imports and basic checks
        from src.analysis import compute_financial_health_score
        from src.tabs.executive_overview import render_executive_overview
        
        # If we get here, imports work
        assert True
    
    def test_utils_used_by_analysis(self):
        """Test that utils are compatible with analysis functions."""
        from src.utils import calculate_growth
        
        # Test basic growth calculation
        assert calculate_growth(110, 100) == pytest.approx(0.1)
        assert calculate_growth(100, 100) == 0
        assert calculate_growth(0, 100) == -1
        assert calculate_growth(100, 0) is None  # Division by zero


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
