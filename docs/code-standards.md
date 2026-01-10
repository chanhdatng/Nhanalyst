# Code Standards & Best Practices

**Version**: 1.0
**Last Updated**: 2025-12-08

---

## Overview

This document defines the coding standards, architectural patterns, and best practices for the Professional Sales Analytics Dashboard project. All contributors must adhere to these guidelines to ensure consistency, maintainability, and code quality.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Python Style Guide](#python-style-guide)
3. [Naming Conventions](#naming-conventions)
4. [Code Organization](#code-organization)
5. [Data Processing Patterns](#data-processing-patterns)
6. [Streamlit Best Practices](#streamlit-best-practices)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Testing Standards](#testing-standards)
10. [Documentation Requirements](#documentation-requirements)
11. [Git Workflow](#git-workflow)

---

## Project Structure

### Directory Layout

```
nhan/
├── .claude/                    # Claude Code configuration
│   └── settings.local.json     # Local permissions
├── src/                        # Core application modules
│   ├── tabs/                   # Dashboard tab modules
│   │   ├── __init__.py        # (Optional) Package initialization
│   │   ├── customer_market.py
│   │   ├── executive_overview.py
│   │   ├── growth_insights.py
│   │   ├── product_intelligence.py
│   │   ├── product_launching.py
│   │   └── vietnam_focus.py
│   ├── analysis.py             # KPI computation & metrics
│   ├── charts.py               # Plotly chart helpers
│   ├── data_processing.py      # Data loading & cleaning
│   ├── ui_helpers.py           # UI components & styling
│   └── utils.py                # Utility functions
├── docs/                       # Documentation
│   ├── codebase-summary.md
│   ├── project-overview-pdr.md
│   ├── code-standards.md       # This file
│   └── system-architecture.md
├── tests/                      # (Future) Test suite
│   ├── test_analysis.py
│   ├── test_data_processing.py
│   └── test_utils.py
├── dashboard.py                # Main application entry
├── debug_active.py             # Debug scripts
├── inspect_data.py             # Utility scripts
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project README
└── data.xlsx                   # Sales data (gitignored)
```

### Module Responsibilities

| Module | Purpose | Exports |
|--------|---------|---------|
| `data_processing.py` | Data ingestion, validation, cleaning | `load_data()`, `clean_data()`, `EXPECTED_COLS` |
| `analysis.py` | Business metrics computation | `compute_top_level_kpis()`, `compute_client_metrics()`, etc. |
| `utils.py` | General utilities | `filter_by_date()`, `calculate_growth()`, `DEFAULT_DATE_COL` |
| `charts.py` | Visualization helpers | `fig_top_level()`, `fig_top_products()`, `fig_region_map()` |
| `ui_helpers.py` | Streamlit UI components | `apply_custom_styles()`, `checkbox_filter()` |
| `tabs/*.py` | Tab rendering logic | `render_*()` functions (one per tab) |
| `dashboard.py` | Application orchestration | `streamlit_app()`, `main()` |

---

## Python Style Guide

### Base Style: PEP 8

Follow [PEP 8](https://peps.python.org/pep-0008/) with the following specific guidelines:

### Line Length

- **Maximum line length**: 120 characters (relaxed from PEP 8's 79)
- Use line breaks for readability, not to enforce strict limits
- Break long function calls using parentheses:

```python
# Good
result = some_function(
    argument1='value1',
    argument2='value2',
    argument3='value3'
)

# Avoid
result = some_function(argument1='value1', argument2='value2', argument3='value3')
```

### Indentation

- **4 spaces** (no tabs)
- Use consistent indentation for continuations

```python
# Good
if (condition1 and condition2 and
    condition3 and condition4):
    do_something()

# Avoid
if (condition1 and condition2 and
  condition3 and condition4):
    do_something()
```

### Imports

- Group imports: standard library, third-party, local
- Use absolute imports for `src` modules
- One import per line

```python
# Good
import argparse
import sys
import datetime as dt

import streamlit as st
import pandas as pd

from src.data_processing import load_data, clean_data
from src.analysis import compute_top_level_kpis

# Avoid
import argparse, sys, datetime as dt  # Multiple imports on one line
from src.data_processing import *     # Wildcard imports
```

### String Formatting

- Use f-strings for string interpolation (Python 3.6+)
- Use `.format()` for complex templates
- Use `%` formatting only for legacy code

```python
# Good
message = f"Total revenue: {revenue:,.2f}"
title = f"{product_name} - {year}"

# Acceptable
message = "Revenue: {:,.2f}".format(revenue)

# Avoid
message = "Revenue: " + str(revenue)
```

### Type Hints (Future Enhancement)

Add type hints for function signatures:

```python
# Recommended (future)
def compute_growth(current: float, previous: float) -> Optional[float]:
    if previous == 0 or pd.isna(previous):
        return None
    return (current - previous) / previous

# Current (acceptable)
def compute_growth(current, previous):
    if previous == 0 or pd.isna(previous):
        return None
    return (current - previous) / previous
```

---

## Naming Conventions

### Variables

- **snake_case** for variables and functions
- Descriptive names (avoid abbreviations unless common)

```python
# Good
total_revenue = df['Sold'].sum()
client_count = df['Name of client'].nunique()
df_filtered = filter_by_date(df, years, months)

# Avoid
tr = df['Sold'].sum()          # Unclear abbreviation
cc = df['Name of client'].nunique()
temp = filter_by_date(df, years, months)  # Generic name
```

### Functions

- **snake_case** for function names
- Use verbs for actions, nouns for getters

```python
# Good
def compute_top_level_kpis(df):
    ...

def filter_by_date(df, years, months):
    ...

# Avoid
def TopLevelKpis(df):  # PascalCase for functions
    ...

def kpis(df):  # Too generic
    ...
```

### Constants

- **UPPER_CASE** with underscores
- Define at module level

```python
# Good
DEFAULT_DATE_COL = 'date__ym'
EXPECTED_COLS = [...]
MIN_SPIKE_THRESHOLD = 0.3

# Avoid
default_date_col = 'date__ym'  # Not uppercase
```

### Classes (Future)

- **PascalCase** for class names
- Descriptive, singular nouns

```python
# Good (future)
class SalesMetricsCalculator:
    def __init__(self, df):
        self.df = df

    def compute_revenue(self):
        return self.df['Sold'].sum()

# Avoid
class sales_metrics_calculator:  # snake_case for class
    ...
```

### DataFrame Columns

- Use existing column names as-is (match source data)
- For derived columns, use **snake_case**

```python
# Good
df['date__ym'] = pd.to_datetime(...)  # Synthetic column
df['rfm_score'] = df['r_score']*100 + df['f_score']*10 + df['m_score']

# Avoid
df['Date'] = pd.to_datetime(...)  # Inconsistent casing
```

### File Naming

- **snake_case** for Python files
- Descriptive names matching module purpose

```python
# Good
data_processing.py
customer_market.py
growth_insights.py

# Avoid
DataProcessing.py  # PascalCase for files
cm.py              # Unclear abbreviation
```

---

## Code Organization

### Module Size

- Keep modules under 500 lines (soft limit)
- Split large modules into sub-modules
- Extract reusable logic into utilities

### Function Size

- Keep functions under 50 lines (soft limit)
- Extract complex logic into helper functions
- One function, one responsibility

```python
# Good
def render_vietnam_focus(df, df_curr, df_prev, has_prev_year, current_year_val):
    df_vn_curr = filter_vietnam_data(df_curr)
    df_vn_prev = filter_vietnam_data(df_prev)

    render_category_focus("Frozen Puree", ["FROZEN PUREE"])
    render_category_focus("Frozen Fruit", ["FROZEN FRUIT"])

def filter_vietnam_data(df):
    # Focused helper function
    vn_mask = df['Country'].str.contains('Viet', case=False, na=False)
    return df[vn_mask]

# Avoid - 200+ line monolithic function
def render_vietnam_focus(df, df_curr, df_prev, has_prev_year, current_year_val):
    # 200 lines of nested logic...
```

### Separation of Concerns

**Layer** | **Responsibility** | **Example Modules**
----------|-------------------|--------------------
Data | Loading, cleaning, validation | `data_processing.py`
Business Logic | KPI computation, metrics | `analysis.py`
Presentation | Charts, UI components | `charts.py`, `ui_helpers.py`
Orchestration | App flow, tab routing | `dashboard.py`, `tabs/*.py`

### Constants and Configuration

- Define constants at module top (after imports)
- Use ALL_CAPS for constants
- Group related constants

```python
# Good
import pandas as pd

# Constants
DEFAULT_DATE_COL = 'date__ym'
SPIKE_THRESHOLD = 0.3
ACTIVE_CUSTOMER_MIN_ORDERS = 2
LOOKBACK_MONTHS = 6

# Functions below...
```

---

## Data Processing Patterns

### DataFrame Operations

#### Prefer Method Chaining

```python
# Good
result = (
    df.groupby('Region')
      .agg({'Sold': 'sum'})
      .sort_values('Sold', ascending=False)
      .reset_index()
)

# Avoid
temp1 = df.groupby('Region').agg({'Sold': 'sum'})
temp2 = temp1.sort_values('Sold', ascending=False)
result = temp2.reset_index()
```

#### Use Explicit Aggregations

```python
# Good
metrics = df.groupby('Name of client').agg(
    revenue=('Sold', 'sum'),
    orders=('Sold', 'count'),
    avg_order=('Sold', 'mean')
).reset_index()

# Avoid
metrics = df.groupby('Name of client')['Sold'].agg(['sum', 'count', 'mean'])
```

#### Handle Missing Values Explicitly

```python
# Good
df['Quantity (KG)'] = pd.to_numeric(df['Quantity (KG)'], errors='coerce').fillna(0.0)

# Avoid
df['Quantity (KG)'] = df['Quantity (KG)'].fillna(0)  # Doesn't handle non-numeric
```

### Data Validation

Always validate input data:

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Validate schema
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = pd.NA  # Add missing column

    # 2. Type coercion with error handling
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # 3. Remove invalid rows
    df = df[~df[DEFAULT_DATE_COL].isna()].reset_index(drop=True)

    return df
```

### Caching Strategy

Use Streamlit caching for expensive operations:

```python
# Good
@st.cache_data(show_spinner=False)
def load_data(file_path_or_buffer, csv_fallback=True, nrows=None):
    # Expensive I/O operation
    df = pd.read_excel(file_path_or_buffer, engine='openpyxl')
    return df

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Expensive cleaning operation
    ...
    return cleaned_df

# Avoid - No caching for repeated operations
def load_data(file_path):
    return pd.read_excel(file_path)
```

---

## Streamlit Best Practices

### UI Components

#### Use Columns for Layout

```python
# Good
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"{revenue:,.0f}")
c2.metric("Volume", f"{volume:,.0f}")
c3.metric("AOV", f"{aov:,.0f}")
c4.metric("Clients", f"{clients:,.0f}")

# Avoid - Vertical stacking
st.metric("Revenue", f"{revenue:,.0f}")
st.metric("Volume", f"{volume:,.0f}")
st.metric("AOV", f"{aov:,.0f}")
st.metric("Clients", f"{clients:,.0f}")
```

#### Use Expanders for Filters

```python
# Good
with st.sidebar.expander("Select Years", expanded=True):
    for year in years:
        st.checkbox(str(year), value=True, key=f"year_{year}")

# Avoid - No grouping
st.sidebar.checkbox("2023", value=True)
st.sidebar.checkbox("2024", value=True)
st.sidebar.checkbox("2025", value=True)
```

#### Handle Empty States

```python
# Good
if df_filtered.empty:
    st.warning("No data available for selected filters. Please adjust your selection.")
    st.stop()

# Continue with non-empty data
st.dataframe(df_filtered)

# Avoid - No validation
st.dataframe(df_filtered)  # Crashes or shows empty table
```

### Session State Management

Use session state for persistent user selections:

```python
# Good (future enhancement)
if 'selected_years' not in st.session_state:
    st.session_state.selected_years = [2024]

# Avoid - Global variables
selected_years = [2024]  # Lost on rerun
```

### Custom Styling

Centralize CSS in `ui_helpers.py`:

```python
def apply_custom_styles():
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #2E3192 0%, #1BFFFF 100%);
        border-radius: 10px;
        padding: 15px 25px;
    }
    </style>
    """, unsafe_allow_html=True)
```

---

## Error Handling

### Defensive Programming

Always validate inputs and handle edge cases:

```python
# Good
def calculate_growth(current_val, prev_val):
    if prev_val == 0 or pd.isna(prev_val):
        return None  # Handle division by zero
    return (current_val - prev_val) / prev_val

# Avoid
def calculate_growth(current_val, prev_val):
    return (current_val - prev_val) / prev_val  # Crashes on zero
```

### Try-Except Blocks

Use specific exceptions:

```python
# Good
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Could not load data. Error: {e}")

# Avoid
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except:  # Bare except
        return None
```

### User-Facing Error Messages

Provide clear, actionable error messages:

```python
# Good
if df_curr.empty:
    st.error("""
    ⚠️ No data found for selected filters.

    **Resolution**:
    1. Check if you've selected at least one Year, Month, and Region
    2. Verify that your data file contains records matching these filters
    3. Review the Data Debug Info in the sidebar for available values
    """)
    st.stop()

# Avoid
if df_curr.empty:
    st.error("No data")  # Not helpful
```

---

## Performance Optimization

### Data Operations

1. **Filter Early**: Reduce dataset size before aggregation

```python
# Good
df_filtered = df[df['Year'].isin(selected_years)]
result = df_filtered.groupby('Region')['Sold'].sum()

# Avoid
result = df.groupby('Region')['Sold'].sum()
result = result[result.index.isin(selected_regions)]  # Late filter
```

2. **Use Vectorized Operations**: Avoid loops on DataFrames

```python
# Good
df['Growth'] = (df['Current'] - df['Baseline']) / df['Baseline']

# Avoid
for idx, row in df.iterrows():
    df.at[idx, 'Growth'] = (row['Current'] - row['Baseline']) / row['Baseline']
```

3. **Limit Data Display**: Use `head()` for large tables

```python
# Good
st.dataframe(df.head(100))  # Show top 100 rows

# Avoid
st.dataframe(df)  # Shows all rows (slow for large datasets)
```

### Streamlit Optimization

1. **Cache Data Loading**: Use `@st.cache_data`
2. **Lazy Tab Rendering**: Tabs render only when selected (built-in)
3. **Avoid Recomputation**: Move expensive operations outside tab renders

---

## Testing Standards

### Unit Tests (Future)

Test individual functions in isolation:

```python
# tests/test_utils.py
import pytest
from src.utils import calculate_growth

def test_calculate_growth_normal():
    assert calculate_growth(120, 100) == 0.2

def test_calculate_growth_zero_baseline():
    assert calculate_growth(100, 0) is None

def test_calculate_growth_negative():
    assert calculate_growth(80, 100) == -0.2
```

### Integration Tests (Future)

Test data pipelines end-to-end:

```python
# tests/test_data_processing.py
def test_load_and_clean_pipeline():
    df_raw = load_data('test_data.xlsx')
    df_clean = clean_data(df_raw)

    assert len(df_clean) > 0
    assert 'date__ym' in df_clean.columns
    assert df_clean['Year'].dtype == 'float64'
```

### Test Coverage Goals

- **Target**: 80% code coverage
- **Priority**: Core logic (analysis, data processing)
- **Tools**: pytest, pytest-cov

---

## Documentation Requirements

### Code Comments

- Comment **why**, not **what**
- Use docstrings for functions

```python
# Good
def compute_rfm_clusters(client_df: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
    """
    Apply K-Means clustering to client RFM features.

    Args:
        client_df: DataFrame with 'recency_days', 'frequency', 'monetary' columns
        n_clusters: Number of clusters (default: 4)

    Returns:
        client_df with 'cluster' column added
    """
    features = client_df[['recency_days', 'frequency', 'monetary']].fillna(0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    client_df['cluster'] = kmeans.fit_predict(X)
    return client_df

# Avoid
def compute_rfm_clusters(client_df, n_clusters=4):
    # Cluster clients
    features = client_df[['recency_days', 'frequency', 'monetary']].fillna(0)
    ...
```

### Inline Comments

Use sparingly for complex logic:

```python
# Good
# Handle edge case: Base volume = 0 indicates new product launch
# Treat as 100% growth for ranking purposes
mask_new = (merged['Base_Vol'] == 0) & (merged['Curr_Vol'] > 0)
merged.loc[mask_new, 'Growth_Pct'] = 1.0

# Avoid
x = 1.0  # Set x to 1.0
```

### README Files

- Every major module should have a docstring at the top
- README.md in root for project overview

---

## Git Workflow

### Branch Naming

- **Feature**: `feature/add-forecasting`
- **Bugfix**: `bugfix/fix-spike-detection`
- **Hotfix**: `hotfix/critical-data-loading`
- **Refactor**: `refactor/modularize-tabs`

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Examples**:
```
feat(analysis): Add customer churn prediction model

Implement logistic regression model to predict customer churn
based on RFM features and order history.

Closes #123
```

```
fix(data_processing): Handle missing Month column gracefully

Previously crashed with KeyError. Now creates synthetic Month=1
if column is missing.

Fixes #456
```

### Pull Request Guidelines

1. **Title**: Clear, descriptive
2. **Description**: What, why, how
3. **Tests**: Add/update tests
4. **Screenshots**: For UI changes
5. **Review**: At least 1 approval required

---

## Code Review Checklist

Before submitting code:

- [ ] Follows PEP 8 style guide
- [ ] No hardcoded values (use constants)
- [ ] Error handling for edge cases
- [ ] Docstrings for new functions
- [ ] No commented-out code
- [ ] Performance optimized (caching, vectorization)
- [ ] Tested manually with sample data
- [ ] Git commit messages follow convention
- [ ] No merge conflicts

---

## Additional Guidelines

### Security

- No hardcoded API keys or credentials
- Use environment variables for secrets
- Validate user inputs to prevent injection

### Accessibility

- Use semantic HTML in Streamlit
- Ensure color contrast (WCAG AA)
- Provide alt text for images

### Internationalization (Future)

- Use constants for UI strings
- Prepare for multi-language support
- Handle date/number formatting by locale

---

## Enforcement

- **Code Reviews**: All PRs require review
- **Linting**: Use `flake8` or `ruff` (future)
- **Pre-commit Hooks**: Auto-format with `black` (future)
- **CI/CD**: Automated tests on push (future)

---

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Pandas Best Practices](https://pandas.pydata.org/docs/user_guide/style.ipynb)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)

---

**Maintained By**: Development Team
**Last Reviewed**: 2025-12-08
**Next Review**: Q1 2026
