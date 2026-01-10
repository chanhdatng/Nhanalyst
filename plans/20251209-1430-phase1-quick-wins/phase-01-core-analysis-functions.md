# Phase 1: Core Analysis Functions

**Phase**: 1 of 3
**Created**: 2025-12-09
**Status**: ✅ COMPLETED
**Priority**: P0 (Blocking Phase 2)
**Duration**: 3-4 days
**Completed**: 2025-12-09

---

## Context

**Parent Plan**: [plan.md](./plan.md)
**Dependencies**: None (foundational phase)
**Related Docs**:
- [src/analysis.py](../../src/analysis.py) - Target file for new functions
- [Code Standards](../../docs/code-standards.md) - Python style guide
- [Codebase Summary](../../docs/codebase-summary.md) - Existing architecture

---

## Overview

Add 5 new computation functions to `src/analysis.py` following existing patterns. All functions cached, defensive error handling, type hints, returns structured dicts/DataFrames.

**Implementation Status**: ✅ Complete (5/5 functions implemented)
**Review Status**: ✅ All 37 tests passing

---

## Key Insights

1. **Existing Pattern Analysis**:
   - Functions return dict or DataFrame
   - Use `DEFAULT_DATE_COL` for temporal operations
   - Leverage `calculate_growth()` utility
   - Simple RFM scoring with quantiles (existing: `compute_rfm_clusters`)

2. **Performance Constraints**:
   - Dataset: 100K-1M rows, daily updates
   - Target: <3s compute time
   - Solution: Caching + efficient pandas operations

3. **Data Availability**:
   - No COGS data → Use AOV as profit margin proxy
   - No marketing spend → Estimate CAC from customer counts
   - Synthetic `date__ym` column for temporal analysis

---

## Requirements

### Functional Requirements

**FR-1**: Financial Health Score
- Input: `df_curr`, `df_prev` (current/previous period DataFrames)
- Output: Dict with score (0-100), color, components breakdown
- Components: Revenue growth (30%), Profit margin proxy (25%), Retention (25%), AOV growth (20%)

**FR-2**: Churn Risk Scores
- Input: `df` (current period DataFrame)
- Output: DataFrame with per-customer churn risk scores (0-100)
- Factors: Recency (40%), Frequency decline (30%), Monetary decline (20%), Variety decline (10%)

**FR-3**: Product Lifecycle Stages
- Input: `df` (full historical DataFrame)
- Output: DataFrame with product lifecycle classification
- Stages: Introduction, Growth, Maturity, Decline

**FR-4**: Growth Decomposition
- Input: `df_curr`, `df_prev`
- Output: Dict with total growth broken into: new customers, expansion, churn, price impact, mix impact

**FR-5**: Launch Velocity
- Input: `df` (full historical), `min_age_months=3`
- Output: DataFrame with velocity scores for products launched <12 months ago
- Formula: `(M3 revenue - M1 revenue) / M1 revenue * 100`

### Non-Functional Requirements

**NFR-1**: Performance
- Each function: <1s for 100K rows, <2s for 1M rows
- Use vectorized pandas operations (no Python loops on rows)

**NFR-2**: Reliability
- Handle empty DataFrames gracefully
- Protect division by zero
- Handle missing/null values

**NFR-3**: Maintainability
- Type hints on all functions
- Docstrings with parameter/return descriptions
- Follow existing code patterns

---

## Architecture

### Function Signatures

```python
@st.cache_data(ttl=3600)
def compute_financial_health_score(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    """
    Compute financial health score (0-100) from 4 weighted components.

    Args:
        df_curr: Current period data
        df_prev: Previous period data (can be empty)

    Returns:
        {
            'score': 75.5,
            'color': 'green',  # red/yellow/green
            'components': {
                'revenue_growth': {'value': 15.2, 'score': 80},
                'profit_margin': {'value': 25.0, 'score': 70},
                'retention': {'value': 85.0, 'score': 85},
                'aov_growth': {'value': 10.5, 'score': 65}
            }
        }
    """

@st.cache_data(ttl=3600)
def compute_churn_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-customer churn risk scores.

    Args:
        df: Full historical data

    Returns:
        DataFrame columns: ['Name of client', 'churn_risk_score', 'risk_level',
                           'days_since_last', 'frequency_trend', 'monetary_trend',
                           'variety_trend', 'total_revenue']
    """

@st.cache_data(ttl=3600)
def compute_product_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify products into lifecycle stages.

    Args:
        df: Full historical data

    Returns:
        DataFrame columns: ['Name of product', 'lifecycle_stage', 'stage_emoji',
                           'age_months', 'growth_rate', 'total_revenue', 'recent_revenue']
    """

@st.cache_data(ttl=3600)
def compute_growth_decomposition(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    """
    Decompose revenue growth into components.

    Args:
        df_curr: Current period data
        df_prev: Previous period data

    Returns:
        {
            'total_growth': 150000,
            'total_growth_pct': 15.0,
            'revenue_prev': 1000000,
            'revenue_curr': 1150000,
            'components': {
                'new_customers': 50000,
                'expansion': 80000,
                'churn': -30000,
                'price_impact': 20000,
                'mix_impact': 30000
            },
            'component_pct': {...}
        }
    """

@st.cache_data(ttl=3600)
def compute_launch_velocity(df: pd.DataFrame, min_age_months: int = 3) -> pd.DataFrame:
    """
    Calculate launch velocity for new products.

    Args:
        df: Full historical data
        min_age_months: Minimum age in months to include

    Returns:
        DataFrame columns: ['Name of product', 'launch_date', 'age_months',
                           'velocity_pct', 'velocity_category', 'velocity_emoji',
                           'm1_revenue', 'm3_revenue', 'current_revenue',
                           'm1_customers', 'm3_customers']
    """
```

### Integration Points

**Existing Code to Reuse**:
- `DEFAULT_DATE_COL` from `src/utils.py`
- `calculate_growth(current, prev)` from `src/utils.py`
- Existing RFM pattern from `compute_rfm_clusters()`
- Existing time aggregation patterns from `compute_top_level_kpis()`

**Data Schema** (from existing code):
```python
Required columns:
- 'date__ym': Synthetic date column (datetime)
- 'Sold': Revenue
- 'Quantity (KG)': Volume
- 'Name of client': Customer identifier
- 'Name of product': Product identifier
- 'Year', 'Month': Temporal dimensions
```

---

## Related Code Files

**Primary**:
- `src/analysis.py` - Add all 5 new functions here

**Dependencies**:
- `src/utils.py` - Import `DEFAULT_DATE_COL`, `calculate_growth`
- `src/data_processing.py` - Understand data schema

**Reference Implementations**:
- `compute_top_level_kpis()` - YoY/MoM growth logic
- `compute_client_metrics()` - RFM scoring pattern
- `compute_product_metrics()` - Product aggregation pattern

---

## Implementation Steps

### Step 1: Add Imports & Setup (15 min)
```python
# At top of src/analysis.py, add:
import streamlit as st
from datetime import timedelta

# Verify existing imports include:
# pandas as pd, numpy as np, sklearn, src.utils
```

### Step 2: Implement `compute_financial_health_score` (2-3 hours)

**Pseudocode**:
```python
def compute_financial_health_score(df_curr, df_prev):
    # Edge case: empty df_prev
    if df_prev.empty:
        df_prev = pd.DataFrame(columns=df_curr.columns)

    # Component 1: Revenue Growth Score
    revenue_curr = df_curr['Sold'].sum()
    revenue_prev = df_prev['Sold'].sum() if not df_prev.empty else 0
    growth_pct = calculate_growth(revenue_curr, revenue_prev) * 100 if revenue_prev > 0 else 0

    # Scoring thresholds
    if growth_pct >= 20:
        revenue_growth_score = 100
    elif growth_pct >= 10:
        revenue_growth_score = 50 + (growth_pct - 10) * 5  # Linear interpolation
    elif growth_pct >= 0:
        revenue_growth_score = 25 + growth_pct * 2.5
    else:
        revenue_growth_score = max(0, 25 + growth_pct)

    # Component 2: Profit Margin Proxy (AOV-based)
    aov_curr = revenue_curr / len(df_curr) if len(df_curr) > 0 else 0
    # Normalize to 0-100 (assume 2000 = excellent AOV)
    profit_margin_score = min(100, (aov_curr / 2000) * 100)

    # Component 3: Customer Retention
    clients_curr = set(df_curr['Name of client'].unique())
    clients_prev = set(df_prev['Name of client'].unique()) if not df_prev.empty else set()

    if clients_prev:
        retained = len(clients_curr & clients_prev)
        retention_rate = (retained / len(clients_prev)) * 100
    else:
        retention_rate = 100

    retention_score = retention_rate

    # Component 4: AOV Growth
    aov_prev = revenue_prev / len(df_prev) if len(df_prev) > 0 else 0
    aov_growth_pct = calculate_growth(aov_curr, aov_prev) * 100 if aov_prev > 0 else 0

    if aov_growth_pct >= 15:
        aov_growth_score = 100
    elif aov_growth_pct >= 5:
        aov_growth_score = 50 + (aov_growth_pct - 5) * 5
    elif aov_growth_pct >= 0:
        aov_growth_score = 25 + aov_growth_pct * 5
    else:
        aov_growth_score = max(0, 25 + aov_growth_pct * 2)

    # Final Score (weighted)
    final_score = (
        revenue_growth_score * 0.30 +
        profit_margin_score * 0.25 +
        retention_score * 0.25 +
        aov_growth_score * 0.20
    )

    # Color determination
    if final_score >= 75:
        color = 'green'
    elif final_score >= 50:
        color = 'yellow'
    else:
        color = 'red'

    return {
        'score': round(final_score, 1),
        'color': color,
        'components': {...}  # Full breakdown
    }
```

**Test Cases**:
- Empty `df_prev` → Should return score based only on current data
- Negative growth → Score should be <50
- Perfect growth (>20% revenue, >15% AOV) → Score should be >90

### Step 3: Implement `compute_churn_risk_scores` (3-4 hours)

**Pseudocode**:
```python
def compute_churn_risk_scores(df):
    today = df['date__ym'].max()
    client_stats = []

    for client in df['Name of client'].unique():
        client_df = df[df['Name of client'] == client]

        # 1. Recency Score (40%)
        last_purchase = client_df['date__ym'].max()
        days_since = (today - last_purchase).days

        if days_since <= 30:
            recency_score = 0
        elif days_since <= 60:
            recency_score = (days_since - 30)
        elif days_since <= 90:
            recency_score = 30 + (days_since - 60)
        else:
            recency_score = min(100, 60 + (days_since - 90) * 0.5)

        # 2. Frequency Decline (30%)
        last_3m = client_df[client_df['date__ym'] >= (today - pd.Timedelta(days=90))]
        prev_3m = client_df[(client_df['date__ym'] >= (today - pd.Timedelta(days=180))) &
                            (client_df['date__ym'] < (today - pd.Timedelta(days=90)))]

        orders_recent = len(last_3m)
        orders_prev = len(prev_3m)

        if orders_prev > 0:
            freq_change = (orders_recent - orders_prev) / orders_prev
            frequency_score = max(0, min(100, (1 - freq_change) * 50))
        else:
            frequency_score = 50

        # 3. Monetary Decline (20%)
        revenue_recent = last_3m['Sold'].sum()
        revenue_prev = prev_3m['Sold'].sum()

        if revenue_prev > 0:
            monetary_change = (revenue_recent - revenue_prev) / revenue_prev
            monetary_score = max(0, min(100, (1 - monetary_change) * 50))
        else:
            monetary_score = 50

        # 4. Variety Decline (10%)
        products_recent = last_3m['Name of product'].nunique()
        products_prev = prev_3m['Name of product'].nunique()

        if products_prev > 0:
            variety_change = (products_recent - products_prev) / products_prev
            variety_score = max(0, min(100, (1 - variety_change) * 50))
        else:
            variety_score = 50

        # Final Churn Risk
        churn_risk = (
            recency_score * 0.40 +
            frequency_score * 0.30 +
            monetary_score * 0.20 +
            variety_score * 0.10
        )

        risk_level = 'High' if churn_risk >= 70 else ('Medium' if churn_risk >= 40 else 'Low')

        client_stats.append({
            'Name of client': client,
            'churn_risk_score': round(churn_risk, 1),
            'risk_level': risk_level,
            'days_since_last': days_since,
            'frequency_trend': round(freq_change * 100, 1) if orders_prev > 0 else None,
            'monetary_trend': round(monetary_change * 100, 1) if revenue_prev > 0 else None,
            'variety_trend': round(variety_change * 100, 1) if products_prev > 0 else None,
            'total_revenue': client_df['Sold'].sum()
        })

    return pd.DataFrame(client_stats).sort_values('churn_risk_score', ascending=False)
```

**Performance Optimization**:
- For >10K customers, implement chunked processing
- Use vectorized operations where possible (groupby instead of loop)

### Step 4: Implement `compute_product_lifecycle` (2-3 hours)

**Pseudocode**:
```python
def compute_product_lifecycle(df):
    today = df['date__ym'].max()
    product_stages = []

    for product in df['Name of product'].unique():
        prod_df = df[df['Name of product'] == product]

        # Calculate age
        first_sale = prod_df['date__ym'].min()
        age_months = ((today - first_sale).days // 30)

        # Calculate revenue
        total_revenue = prod_df['Sold'].sum()

        # Calculate growth rate (last 3m vs prev 3m)
        last_3m = prod_df[prod_df['date__ym'] >= (today - pd.Timedelta(days=90))]
        prev_3m = prod_df[(prod_df['date__ym'] >= (today - pd.Timedelta(days=180))) &
                          (prod_df['date__ym'] < (today - pd.Timedelta(days=90)))]

        revenue_recent = last_3m['Sold'].sum()
        revenue_prev = prev_3m['Sold'].sum()

        if revenue_prev > 0:
            growth_rate = (revenue_recent - revenue_prev) / revenue_prev
        else:
            growth_rate = 0 if revenue_recent == 0 else 1

        # Lifecycle stage logic
        if age_months < 6 and total_revenue < 10000 and growth_rate > 0.5:
            stage, emoji = 'Introduction', '🌱'
        elif age_months <= 18 and growth_rate > 0.2:
            stage, emoji = 'Growth', '📈'
        elif age_months > 18 and abs(growth_rate) <= 0.1:
            stage, emoji = 'Maturity', '💎'
        elif growth_rate < -0.1:
            # Check for 3 consecutive months decline
            monthly = prod_df.groupby(prod_df['date__ym'].dt.to_period('M'))['Sold'].sum()
            if len(monthly) >= 3 and all(monthly.tail(3).diff().dropna() < 0):
                stage, emoji = 'Decline', '📉'
            else:
                stage, emoji = 'Maturity', '💎'
        else:
            stage, emoji = 'Maturity', '💎'

        product_stages.append({
            'Name of product': product,
            'lifecycle_stage': stage,
            'stage_emoji': emoji,
            'age_months': age_months,
            'growth_rate': round(growth_rate * 100, 1),
            'total_revenue': total_revenue,
            'recent_revenue': revenue_recent
        })

    return pd.DataFrame(product_stages)
```

### Step 5: Implement `compute_growth_decomposition` (3-4 hours)

**Pseudocode**:
```python
def compute_growth_decomposition(df_curr, df_prev):
    if df_prev.empty:
        return None

    revenue_curr = df_curr['Sold'].sum()
    revenue_prev = df_prev['Sold'].sum()
    total_growth = revenue_curr - revenue_prev

    # Customer sets
    clients_curr = set(df_curr['Name of client'].unique())
    clients_prev = set(df_prev['Name of client'].unique())

    # 1. New Customer Revenue
    new_clients = clients_curr - clients_prev
    new_customer_revenue = df_curr[df_curr['Name of client'].isin(new_clients)]['Sold'].sum()

    # 2. Churned Customer Revenue
    churned_clients = clients_prev - clients_curr
    churn_revenue = -df_prev[df_prev['Name of client'].isin(churned_clients)]['Sold'].sum()

    # 3. Existing Customer Expansion
    retained_clients = clients_curr & clients_prev
    existing_revenue_curr = df_curr[df_curr['Name of client'].isin(retained_clients)]['Sold'].sum()
    existing_revenue_prev = df_prev[df_prev['Name of client'].isin(retained_clients)]['Sold'].sum()
    expansion_revenue = existing_revenue_curr - existing_revenue_prev

    # 4. Price Impact
    avg_price_curr = df_curr['Sold'].sum() / df_curr['Quantity (KG)'].sum()
    avg_price_prev = df_prev['Sold'].sum() / df_prev['Quantity (KG)'].sum()
    volume_curr = df_curr['Quantity (KG)'].sum()
    price_impact = (avg_price_curr - avg_price_prev) * volume_curr

    # 5. Mix Impact (residual)
    explained = new_customer_revenue + churn_revenue + expansion_revenue + price_impact
    mix_impact = total_growth - explained

    return {
        'total_growth': round(total_growth, 0),
        'total_growth_pct': round((total_growth / revenue_prev) * 100, 1),
        'revenue_prev': round(revenue_prev, 0),
        'revenue_curr': round(revenue_curr, 0),
        'components': {
            'new_customers': round(new_customer_revenue, 0),
            'expansion': round(expansion_revenue, 0),
            'churn': round(churn_revenue, 0),
            'price_impact': round(price_impact, 0),
            'mix_impact': round(mix_impact, 0)
        }
    }
```

### Step 6: Implement `compute_launch_velocity` (2-3 hours)

**Pseudocode**:
```python
def compute_launch_velocity(df, min_age_months=3):
    today = df['date__ym'].max()
    twelve_months_ago = today - pd.Timedelta(days=365)

    product_launches = []

    for product in df['Name of product'].unique():
        prod_df = df[df['Name of product'] == product]
        launch_date = prod_df['date__ym'].min()

        # Only products launched in last 12 months
        if launch_date < twelve_months_ago:
            continue

        age_months = ((today - launch_date).days // 30)

        # Need at least min_age_months data
        if age_months < min_age_months:
            continue

        # M1, M3 revenue
        m1_end = launch_date + pd.Timedelta(days=30)
        m3_end = launch_date + pd.Timedelta(days=90)

        m1_df = prod_df[(prod_df['date__ym'] >= launch_date) & (prod_df['date__ym'] < m1_end)]
        m1_revenue = m1_df['Sold'].sum()

        m3_df = prod_df[(prod_df['date__ym'] >= launch_date) & (prod_df['date__ym'] < m3_end)]
        m3_revenue = m3_df['Sold'].sum()

        # Velocity calculation
        if m1_revenue > 0:
            velocity = ((m3_revenue - m1_revenue) / m1_revenue) * 100
        else:
            velocity = 0 if m3_revenue == 0 else 999

        # Categorize
        if velocity >= 100:
            category, emoji = 'Fast', '🚀'
        elif velocity >= 50:
            category, emoji = 'Moderate', '🏃'
        else:
            category, emoji = 'Slow', '🐌'

        product_launches.append({
            'Name of product': product,
            'launch_date': launch_date,
            'age_months': age_months,
            'velocity_pct': round(velocity, 1),
            'velocity_category': category,
            'velocity_emoji': emoji,
            'm1_revenue': round(m1_revenue, 0),
            'm3_revenue': round(m3_revenue, 0),
            'current_revenue': round(prod_df['Sold'].sum(), 0),
            'm1_customers': m1_df['Name of client'].nunique(),
            'm3_customers': m3_df['Name of client'].nunique()
        })

    return pd.DataFrame(product_launches).sort_values('velocity_pct', ascending=False)
```

### Step 7: Add Unit Tests (2-3 hours)

Create `tests/test_phase1_functions.py`:

```python
import pytest
import pandas as pd
from src.analysis import (
    compute_financial_health_score,
    compute_churn_risk_scores,
    compute_product_lifecycle,
    compute_growth_decomposition,
    compute_launch_velocity
)

def create_mock_df(revenue=100000, clients=10, products=5):
    """Helper to create mock data"""
    return pd.DataFrame({
        'date__ym': pd.date_range('2024-01-01', periods=12, freq='M'),
        'Sold': [revenue/12] * 12,
        'Quantity (KG)': [1000] * 12,
        'Name of client': [f'Client {i%clients}' for i in range(12)],
        'Name of product': [f'Product {i%products}' for i in range(12)],
        'Year': [2024] * 12,
        'Month': list(range(1, 13))
    })

def test_financial_health_score_empty_prev():
    df_curr = create_mock_df()
    df_prev = pd.DataFrame()
    result = compute_financial_health_score(df_curr, df_prev)
    assert result['score'] >= 0 and result['score'] <= 100
    assert result['color'] in ['red', 'yellow', 'green']

def test_churn_risk_scores():
    df = create_mock_df()
    result = compute_churn_risk_scores(df)
    assert 'churn_risk_score' in result.columns
    assert result['churn_risk_score'].between(0, 100).all()

# Add more tests...
```

---

## Todo List

- [x] **Setup**: Add imports to `src/analysis.py`
- [x] **Feature 1**: Implement `compute_financial_health_score()`
- [x] **Feature 1**: Test with mock data (empty prev, negative growth, high growth)
- [x] **Feature 2**: Implement `compute_churn_risk_scores()`
- [x] **Feature 2**: Test with edge cases (new customers, inactive customers)
- [x] **Feature 3**: Implement `compute_product_lifecycle()`
- [x] **Feature 3**: Test lifecycle transitions
- [x] **Feature 4**: Implement `compute_growth_decomposition()`
- [x] **Feature 4**: Validate waterfall math (components sum to total)
- [x] **Feature 5**: Implement `compute_launch_velocity()`
- [x] **Feature 5**: Test with products <3 months old (should skip)
- [x] **Testing**: Create `tests/test_phase1_functions.py`
- [x] **Testing**: Run all unit tests, achieve >70% coverage (37/37 tests passing)
- [x] **Performance**: Test with 100K row dataset, verify <1s per function
- [x] **Code Review**: Self-review for code standards compliance
- [x] **Documentation**: Add docstrings to all functions

---

## Success Criteria

### Functional
- ✅ All 5 functions return expected data structures
- ✅ Edge cases handled (empty data, division by zero, missing values)
- ✅ Results mathematically correct (spot-check calculations)

### Performance
- ✅ Each function <1s for 100K rows
- ✅ Caching works (2nd call instant)

### Code Quality
- ✅ Type hints on all functions
- ✅ Docstrings with Args/Returns
- ✅ Follows PEP 8 style guide
- ✅ No linting errors

### Testing
- ✅ Unit tests for all functions
- ✅ Edge case coverage
- ✅ Code coverage >70%

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance degradation | High | Profile with `%timeit`, optimize pandas ops |
| Complex churn logic errors | Medium | Extensive unit tests, manual validation |
| Lifecycle stage misclassification | Low | Clear thresholds, test with known products |
| Growth decomposition doesn't sum | Medium | Add assertion: `sum(components) == total_growth` |

---

## Security Considerations

- No user input directly used in calculations
- No file writes or external API calls
- Cache TTL prevents stale data (1 hour)
- Defensive programming for null/empty data

---

## Next Steps

1. **Start Implementation**: Begin with Feature 1 (Financial Health Score)
2. **Incremental Testing**: Test each function immediately after implementation
3. **Daily Check-in**: Review progress, blockers
4. **Complete Phase 1**: All functions implemented, tested, passing
5. **Proceed to Phase 2**: UI Integration (blocked until Phase 1 done)
