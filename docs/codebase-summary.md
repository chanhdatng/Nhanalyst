# Codebase Summary

Generated: 2025-12-08

## Overview

This is a professional **Sales Analytics Dashboard** application built with Python and Streamlit. The system provides comprehensive business intelligence for sales performance analysis, including revenue tracking, customer segmentation, product intelligence, and regional market insights with a specific focus on Vietnam operations.

## Project Statistics

- **Total Files**: 17 source files
- **Total Tokens**: 24,418 tokens
- **Total Characters**: 108,909 chars
- **Primary Language**: Python
- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: scikit-learn (K-Means clustering)

## Top 5 Files by Complexity

1. **src/tabs/product_launching.py** (5,593 tokens, 27,163 chars, 22.9%)
   - Product launch analysis and tracking
   - Active customer identification
   - 6-month gap analysis

2. **src/tabs/growth_insights.py** (4,564 tokens, 22,421 chars, 18.7%)
   - Spike detection algorithms
   - YoY growth analysis
   - Client segmentation (new/lost/existing)

3. **dashboard.py** (2,516 tokens, 10,799 chars, 10.3%)
   - Main application entry point
   - Streamlit app orchestration
   - CLI interface

4. **src/tabs/vietnam_focus.py** (2,106 tokens, 9,579 chars, 8.6%)
   - Vietnam-specific market analysis
   - Regional performance tracking
   - Category focus analysis

5. **src/data_processing.py** (1,570 tokens, 6,571 chars, 6.4%)
   - Data loading and cleaning
   - Schema validation
   - Date normalization

## Directory Structure

```
.
├── .claude/                    # Claude Code configuration
│   └── settings.local.json
├── src/                        # Core application modules
│   ├── tabs/                   # Dashboard tab modules
│   │   ├── customer_market.py  # Customer & market analysis
│   │   ├── executive_overview.py # Executive KPI dashboard
│   │   ├── growth_insights.py  # Growth & spike analysis
│   │   ├── product_intelligence.py # Product performance
│   │   ├── product_launching.py # Product launch tracking
│   │   └── vietnam_focus.py    # Vietnam market focus
│   ├── analysis.py             # KPI computation & metrics
│   ├── charts.py               # Plotly chart helpers
│   ├── data_processing.py      # Data loading & cleaning
│   ├── ui_helpers.py           # UI components & styling
│   └── utils.py                # Utility functions
├── dashboard.py                # Main application entry
├── debug_active.py             # Active customer debugging
├── inspect_data.py             # Data inspection utility
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── data.xlsx                   # Sales data (gitignored)
```

## Core Components

### 1. Data Layer (`src/data_processing.py`)

**Purpose**: Robust data ingestion, validation, and normalization

**Key Functions**:
- `load_data(file_path_or_buffer, csv_fallback, nrows)`: Load Excel/CSV with multi-sheet support
- `clean_data(df)`: Standardize schema, handle missing values, create synthetic date column

**Data Schema**:
```python
EXPECTED_COLS = [
    'Year', 'Month', 'Name of client', 'Channel by Sales Person',
    'Region', 'Country', 'Name of product', 'Kind of fruit', 'SKU',
    'Type of product', 'Sold', 'Quantity (KG)'
]
```

**Data Cleaning Steps**:
1. Fuzzy column mapping for header variations
2. Numeric coercion with error handling
3. Text standardization (strip, title-case)
4. Product name normalization (remove 'ANDROS PROFESSIONAL')
5. Synthetic date column creation (`date__ym` = 1st of month)
6. Missing value imputation

**Caching Strategy**: Uses `@st.cache_data` for performance optimization

### 2. Analysis Layer (`src/analysis.py`)

**Purpose**: Compute business metrics and KPIs

**Key Functions**:

1. `compute_top_level_kpis(df)` → dict
   - Total revenue, total KG, active clients
   - YoY/MoM growth calculations
   - New vs churned client analysis
   - Top product/fruit identification

2. `compute_client_metrics(df)` → DataFrame
   - Revenue, KG, orders per client
   - RFM scoring (Recency, Frequency, Monetary)
   - Customer lifetime value approximation

3. `compute_product_metrics(df)` → DataFrame
   - Revenue, KG, orders per product/SKU
   - Price per KG calculation
   - Product performance ranking

4. `compute_region_metrics(df)` → DataFrame
   - Revenue and volume by Region/Country

5. `compute_rfm_clusters(client_df, n_clusters=4)` → DataFrame
   - K-Means clustering on RFM features
   - MinMax scaling for normalization

**Metrics Glossary**:
- **Recency**: Days since last order
- **Frequency**: Total order count
- **Monetary**: Total revenue
- **RFM Score**: Composite score (R×100 + F×10 + M)
- **YoY Growth**: (Current Year - Previous Year) / Previous Year
- **MoM Growth**: (Current Month - Previous Month) / Previous Month

### 3. Visualization Layer (`src/charts.py`)

**Purpose**: Plotly chart generation helpers

**Chart Types**:
- Line charts: Monthly revenue trends
- Bar charts: Top products, regional performance
- Choropleth maps: Revenue by country (with fallback)

**Template**: `plotly_white` for professional appearance

### 4. UI Layer (`src/ui_helpers.py`)

**Purpose**: Streamlit UI components and custom styling

**Components**:

1. `apply_custom_styles()`: Injects custom CSS
   - Metric cards: Dark blue → Cyan gradient
   - Tab styling: Professional rounded tabs
   - Color palette: Professional blue tones

2. `checkbox_filter(label, options, key_prefix, default_selected, expanded)`: Multi-select filter
   - Checkbox-based filtering
   - Collapsible expanders
   - Default selection support

**Design System**:
- Primary: #1E90FF (Dodger Blue)
- Secondary: #D3D3D3 (Light Gray)
- Gradient: #2E3192 → #1BFFFF (Dark Blue → Cyan)

### 5. Utilities (`src/utils.py`)

**Key Functions**:
- `filter_by_date(df, years, months)`: Multi-dimensional date filtering
- `calculate_growth(current_val, prev_val)`: Safe growth calculation
- `ai_insights_summary(...)`: AI-powered insights (OpenAI integration)
- `export_reports(...)`: CSV/JSON export functionality

**Constants**:
- `DEFAULT_DATE_COL = 'date__ym'`: Synthetic date column name

### 6. Dashboard Tabs (`src/tabs/`)

#### 6.1 Executive Overview (`executive_overview.py`)

**Features**:
- KPI cards: Revenue, Volume, AOV, Active Clients
- YoY/MoM comparison with growth indicators
- Monthly revenue trend charts (Bar/Line toggle)
- Multi-year support with distinct series

**Comparison Logic**:
- Single year mode: YoY if prev year exists, else MoM
- Multi-year mode: Comparative trend lines

#### 6.2 Product Intelligence (`product_intelligence.py`)

**Features**:
- Product performance table with contribution %
- Product comparison tool (select multiple products)
- Performance by Type of Product analysis

**Interactive Elements**:
- Multi-select for product comparison
- Trend visualization over months

#### 6.3 Customer & Market (`customer_market.py`)

**Features**:
- Regional performance: Revenue by Country (horizontal bar)
- Client segmentation: Diamond/Gold/Silver/Bronze (RFM-based)
- Top clients table with 6-month gap analysis

**Segmentation Thresholds**:
- Diamond: > 50,000 KG
- Gold: > 10,000 KG
- Silver: > 1,000 KG
- Bronze: ≤ 1,000 KG

**Gap Analysis**:
- Types of Product purchased (last 6m)
- Fruit variety gaps: ✅ bought vs ❌ missing

#### 6.4 Growth & Insights (`growth_insights.py`)

**Features**:

1. **Product Type Performance**:
   - Volume by Type of Product
   - Fruit variety count
   - Monthly trend for top 5 types

2. **Spike Detection**:
   - Growth > 30% threshold
   - YoY or MoM comparison modes
   - Minimum volume filter
   - Client drill-down for spiked SKUs

3. **YoY Growth Drivers**:
   - New clients contribution
   - Lost clients impact
   - Existing clients expansion/contraction
   - Client detail lists with volume changes

**Spike Detection Algorithm**:
```python
Growth % = (Current Volume - Baseline Volume) / Baseline Volume
Spike = Growth > 30% AND Current Volume >= Min Threshold
```

#### 6.5 Vietnam Focus (`vietnam_focus.py`)

**Features**:
- Country filter: 'Viet Nam' (case-insensitive)
- Category focus analysis (custom keyword filters)
- Top 10 products with YoY comparison
- Regional breakdown: South/North/Center (100% stacked bar)
- AI-powered insights for each category

**Regional Mapping**:
- South: Green (#2ca02c)
- North: Blue (#1f77b4)
- Center: Orange (#ff7f0e)

#### 6.6 Product Launching (`product_launching.py`)

**Features**:

1. **Launch Analysis**:
   - Filter by Type of Product and Kind of Fruit
   - Identify new products (not in previous year)
   - Active customer tracking (≥ 2 orders in 6 months)

2. **Launch Table**:
   - Group by Type and/or Kind
   - Customer count with growth indicators
   - Active customer ratio
   - Interactive drill-down dialog

3. **Customer Journey Analysis**:
   - Top customers by volume
   - Last purchase tracking
   - Customer retention metrics

**Active Customer Definition**:
- ≥ 2 orders in last 6 months for a specific Product Type/Kind

### 7. Main Application (`dashboard.py`)

**Entry Points**:

1. **CLI Mode** (`--mode kpis`):
   ```bash
   python dashboard.py --file sales.xlsx --mode kpis
   ```
   - Computes KPIs
   - Exports CSV/JSON reports
   - Prints summary to console

2. **Streamlit Mode** (`--mode streamlit`):
   ```bash
   streamlit run dashboard.py -- --file sales.xlsx --mode streamlit
   ```
   - Launches interactive web dashboard
   - File uploader if no CLI file provided

**Global Filters** (Sidebar):
- Year (checkbox, default: current system year)
- Month (checkbox, default: all)
- Region (checkbox, default: all, expanded)
- Channel by Sales Person (checkbox if exists)
- Country (checkbox if exists)

**Tabs**:
1. Executive Overview
2. Product Intelligence
3. Customer & Market
4. Growth & Insights
5. Vietnam Focus
6. Product Launching

**Data Flow**:
1. Load raw data → Clean data → Apply filters
2. Compute current period dataset (`df_curr`)
3. Compute comparison period dataset (`df_prev`) if single year mode
4. Render tabs with filtered data

## Dependencies

### Core Libraries
```
pandas          # Data manipulation
openpyxl        # Excel file support
plotly          # Interactive visualizations
streamlit       # Web dashboard framework
scikit-learn    # ML clustering
numpy           # Numerical operations
```

### Optional
- `openai`: AI insights generation (if API key provided)

## Configuration

### Claude Code Permissions
```json
{
  "permissions": {
    "allow": [
      "Bash(test:*)",
      "Bash(mkdir:*)",
      "Bash(npx repomix:*)"
    ]
  }
}
```

## Data File Structure

**Expected Format**: Excel (.xlsx) or CSV with headers

**Sample Structure**:
```
Year | Month | Name of client | Channel by Sales Person | Region | Country | Name of product | Kind of fruit | SKU | Type of product | Sold | Quantity (KG)
2024 | 1     | Client A       | Retail                  | South  | Viet Nam| Apple Puree     | Apple         | A01 | FROZEN PUREE    | 100  | 100
```

**Multi-Sheet Support**: All sheets are concatenated during load

**Data Cleaning Notes**:
- 'Sold' column defaults to 'Quantity (KG)' if not found
- 'ANDROS PROFESSIONAL' is stripped from product names
- Month values can be strings (e.g., 'R1', 'M01') - digits are extracted
- Missing clients default to 'Unknown client'
- Text columns are title-cased for consistency

## Key Algorithms

### 1. RFM Clustering
```python
# Features: Recency (days), Frequency (orders), Monetary (revenue)
# Normalization: MinMaxScaler
# Algorithm: K-Means (k=4)
# Output: cluster column (0-3)
```

### 2. Growth Calculation
```python
def calculate_growth(current, previous):
    if previous == 0 or isnan(previous):
        return None
    return (current - previous) / previous
```

### 3. Spike Detection
```python
# 1. Group by SKU: Current vs Baseline
# 2. Calculate Growth %
# 3. Handle edge cases:
#    - Base=0, Curr>0: 100% growth (new listing)
#    - Infinite/NaN: Replace with 0
# 4. Filter: Growth > 30% AND Current Volume >= Min Threshold
# 5. Sort by Growth % descending
```

### 4. Active Customer Detection (6-Month)
```python
# 1. Filter last 6 months of data
# 2. Group by Product/Client
# 3. Count orders
# 4. Active = Order Count >= 2
```

## Performance Optimizations

1. **Caching**: `@st.cache_data` on `load_data()` and `clean_data()`
2. **Lazy Loading**: Tabs render only when selected
3. **Streamlit State**: Filters use session state for reactivity
4. **Data Aggregation**: Pre-aggregated metrics to reduce computation

## Debug & Utility Scripts

### `debug_active.py`
- Purpose: Debug active customer calculation
- Computes 6-month active pairs (Product × Client)
- Checks for logic errors in active customer detection

### `inspect_data.py`
- Purpose: Quick data inspection
- Prints columns, dtypes, and first 3 rows
- Used for schema validation

## Security Considerations

1. **.gitignore**: Excludes data files (*.xlsx), virtual env, pycache
2. **No hardcoded credentials**: OpenAI key passed via CLI arg
3. **Input validation**: Numeric coercion with error handling
4. **File path safety**: Uses pathlib for cross-platform compatibility

## Error Handling Patterns

1. **File Loading**: Try Excel → Fallback to CSV → Raise ValueError
2. **Missing Columns**: Create placeholders with pd.NA
3. **Date Parsing**: Try-except with fallback to pd.NaT
4. **Division by Zero**: Replace with 0 or np.nan
5. **Empty DataFrames**: Early return or display warning

## Testing & Validation

**Manual Testing**:
- Data debug info in sidebar (row count, columns, sample data)
- Unique years/regions display for validation

**Recommended Testing**:
- Unit tests for `compute_*` functions
- Integration tests for data pipeline
- UI tests for Streamlit components

## Future Enhancements (Inferred)

1. **AI Insights**: Full integration with OpenAI for automated insights
2. **Forecasting**: Time-series prediction for revenue
3. **Anomaly Detection**: Advanced spike detection with ML
4. **Export Formats**: PDF reports, PowerPoint slides
5. **Real-time Data**: Database integration (PostgreSQL, MongoDB)
6. **User Authentication**: Multi-user access with roles
7. **Advanced RFM**: Customizable segmentation thresholds

## Code Quality Notes

**Strengths**:
- Modular architecture with clear separation of concerns
- Comprehensive data cleaning and validation
- Professional UI/UX with custom styling
- Extensive business logic for sales analytics

**Areas for Improvement**:
- Add type hints for better IDE support
- Extract magic numbers to constants
- Add docstrings to all functions
- Implement unit tests
- Extract hardcoded thresholds to config file
- Refactor large functions (e.g., `render_product_launching`)

## Deployment Considerations

**Local Development**:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

**Production Deployment**:
- Streamlit Cloud: Direct GitHub integration
- Docker: Containerize with Dockerfile
- Cloud platforms: AWS/GCP/Azure with gunicorn

**Environment Variables**:
- `OPENAI_API_KEY`: For AI insights
- `DATA_FILE_PATH`: Default data file location

## Conclusion

This codebase represents a **production-ready sales analytics platform** with sophisticated business intelligence capabilities. The modular architecture, comprehensive data processing, and interactive visualizations make it suitable for enterprise sales teams requiring detailed performance insights. The Vietnam-specific focus and product launching analysis demonstrate customization for specific business needs.

**Key Strengths**: Robust data handling, professional UI, comprehensive analytics, modular design
**Primary Use Case**: Sales performance tracking, customer segmentation, market analysis, product launch monitoring
