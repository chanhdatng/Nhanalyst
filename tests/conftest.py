"""
Pytest fixtures for Nhanalyst test suite.

Provides reusable test fixtures for mock data generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def mock_df_small():
    """Small dataset (100 rows) for quick tests."""
    np.random.seed(42)  # Reproducible results
    dates = pd.date_range('2024-01-01', periods=12, freq='ME')
    data = []
    for i in range(100):
        data.append({
            'date__ym': dates[i % 12],
            'Sold': np.random.randint(1000, 10000),
            'Quantity (KG)': np.random.randint(100, 1000),
            'Name of client': f'Client {i % 10}',
            'Name of product': f'Product {i % 5}',
            'Year': 2024,
            'Month': (i % 12) + 1,
            'SKU': f'SKU{i % 5}',
            'Kind of fruit': f'Fruit {i % 3}',
            'Type of product': f'Type {i % 2}',
            'Region': 'US',
            'Country': 'United States',
            'Channel by Sales Person': f'Channel {i % 2}'
        })
    return pd.DataFrame(data)


@pytest.fixture
def mock_df_large():
    """Large dataset (100K rows) for performance tests."""
    np.random.seed(42)
    rows = 100000
    dates = pd.date_range('2022-01-01', periods=36, freq='ME')
    data = {
        'date__ym': np.random.choice(dates, rows),
        'Sold': np.random.randint(100, 50000, rows),
        'Quantity (KG)': np.random.randint(10, 5000, rows),
        'Name of client': [f'Client {i}' for i in np.random.randint(0, 1000, rows)],
        'Name of product': [f'Product {i}' for i in np.random.randint(0, 100, rows)],
        'Year': np.random.choice([2022, 2023, 2024], rows),
        'Month': np.random.randint(1, 13, rows),
        'SKU': [f'SKU{i}' for i in np.random.randint(0, 100, rows)],
        'Kind of fruit': [f'Fruit {i}' for i in np.random.randint(0, 10, rows)],
        'Type of product': [f'Type {i}' for i in np.random.randint(0, 5, rows)],
        'Region': np.random.choice(['US', 'EU', 'APAC'], rows),
        'Country': np.random.choice(['United States', 'Germany', 'Japan'], rows),
        'Channel by Sales Person': np.random.choice(['Channel A', 'Channel B'], rows)
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_df_empty():
    """Empty DataFrame with correct schema."""
    return pd.DataFrame(columns=[
        'date__ym', 'Sold', 'Quantity (KG)', 'Name of client',
        'Name of product', 'Year', 'Month', 'SKU', 'Kind of fruit',
        'Type of product', 'Region', 'Country', 'Channel by Sales Person'
    ])


@pytest.fixture
def mock_df_edge_cases():
    """DataFrame with edge cases (nulls, zeros, extremes)."""
    return pd.DataFrame({
        'date__ym': pd.date_range('2024-01-01', periods=10, freq='ME'),
        'Sold': [0, 1, None, 1000000, -100, 50, 50, 50, 50, 50],
        'Quantity (KG)': [0, 1, 1, 100000, 1, None, 10, 10, 10, 10],
        'Name of client': ['Client A'] * 10,
        'Name of product': ['Product X'] * 10,
        'Year': [2024] * 10,
        'Month': list(range(1, 11)),
        'SKU': ['SKU1'] * 10,
        'Kind of fruit': ['Fruit A'] * 10,
        'Type of product': ['Type A'] * 10,
        'Region': ['US'] * 10,
        'Country': ['United States'] * 10,
        'Channel by Sales Person': ['Channel A'] * 10
    })


@pytest.fixture
def mock_df_curr_prev():
    """Fixture providing current and previous period DataFrames for comparison."""
    np.random.seed(42)
    
    # Previous year data
    dates_prev = pd.date_range('2023-01-01', periods=12, freq='ME')
    data_prev = []
    for i in range(60):
        data_prev.append({
            'date__ym': dates_prev[i % 12],
            'Sold': np.random.randint(800, 8000),
            'Quantity (KG)': np.random.randint(80, 800),
            'Name of client': f'Client {i % 8}',  # 8 clients
            'Name of product': f'Product {i % 5}',
            'Year': 2023,
            'Month': (i % 12) + 1,
            'SKU': f'SKU{i % 5}',
            'Kind of fruit': f'Fruit {i % 3}',
            'Type of product': f'Type {i % 2}'
        })
    df_prev = pd.DataFrame(data_prev)
    
    # Current year data (slightly more revenue, some new clients)
    dates_curr = pd.date_range('2024-01-01', periods=12, freq='ME')
    data_curr = []
    for i in range(80):
        data_curr.append({
            'date__ym': dates_curr[i % 12],
            'Sold': np.random.randint(1000, 12000),  # Higher revenue
            'Quantity (KG)': np.random.randint(100, 1000),
            'Name of client': f'Client {i % 10}',  # 10 clients (2 new)
            'Name of product': f'Product {i % 6}',  # 6 products (1 new)
            'Year': 2024,
            'Month': (i % 12) + 1,
            'SKU': f'SKU{i % 6}',
            'Kind of fruit': f'Fruit {i % 3}',
            'Type of product': f'Type {i % 2}'
        })
    df_curr = pd.DataFrame(data_curr)
    
    return df_curr, df_prev


def generate_large_dataset(rows: int = 100000) -> pd.DataFrame:
    """
    Generate realistic large dataset for performance testing.
    
    Args:
        rows: Number of rows to generate
    
    Returns:
        DataFrame with sales data
    """
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=36, freq='ME')
    data = {
        'date__ym': np.random.choice(dates, rows),
        'Sold': np.random.randint(100, 50000, rows).astype(float),
        'Quantity (KG)': np.random.randint(10, 5000, rows).astype(float),
        'Name of client': [f'Client {i}' for i in np.random.randint(0, 1000, rows)],
        'Name of product': [f'Product {i}' for i in np.random.randint(0, 100, rows)],
        'Year': np.random.choice([2022, 2023, 2024], rows),
        'Month': np.random.randint(1, 13, rows),
        'SKU': [f'SKU{i}' for i in np.random.randint(0, 100, rows)],
        'Kind of fruit': [f'Fruit {i}' for i in np.random.randint(0, 10, rows)],
        'Type of product': [f'Type {i}' for i in np.random.randint(0, 5, rows)],
        'Region': np.random.choice(['US', 'EU', 'APAC'], rows),
        'Country': np.random.choice(['United States', 'Germany', 'Japan'], rows),
        'Channel by Sales Person': np.random.choice(['Channel A', 'Channel B'], rows)
    }
    return pd.DataFrame(data)
