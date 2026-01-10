# System Architecture

**Project**: Professional Sales Analytics Dashboard
**Version**: 1.0
**Last Updated**: 2025-12-08

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Data Architecture](#data-architecture)
4. [Application Architecture](#application-architecture)
5. [UI Architecture](#ui-architecture)
6. [Deployment Architecture](#deployment-architecture)
7. [Security Architecture](#security-architecture)
8. [Performance Considerations](#performance-considerations)
9. [Scalability & Future State](#scalability--future-state)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│                    (Desktop/Tablet/Mobile)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Web Server                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   dashboard.py                            │  │
│  │              (Application Orchestrator)                   │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────┴───────────────────────────────────┐  │
│  │              UI Layer (src/tabs/*.py)                     │  │
│  │  ┌────────────┬────────────┬────────────┬────────────┐   │  │
│  │  │ Executive  │  Product   │ Customer   │  Growth    │   │  │
│  │  │  Overview  │Intelligence│  & Market  │  Insights  │   │  │
│  │  └────────────┴────────────┴────────────┴────────────┘   │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────┴───────────────────────────────────┐  │
│  │          Business Logic Layer (src/analysis.py)           │  │
│  │    ┌────────────┬────────────┬────────────┬──────────┐   │  │
│  │    │ KPI Comp.  │ Client RFM │  Product   │  Region  │   │  │
│  │    │            │ Clustering │  Metrics   │  Metrics │   │  │
│  │    └────────────┴────────────┴────────────┴──────────┘   │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────┴───────────────────────────────────┐  │
│  │       Data Processing Layer (src/data_processing.py)      │  │
│  │    ┌────────────┬────────────┬────────────┬──────────┐   │  │
│  │    │   Load     │   Clean    │  Validate  │  Cache   │   │  │
│  │    │   Data     │   Data     │   Schema   │  Data    │   │  │
│  │    └────────────┴────────────┴────────────┴──────────┘   │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
└─────────────────────────┼────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   Data Sources      │
              │  ┌───────────────┐  │
              │  │  Excel/CSV    │  │
              │  │  (data.xlsx)  │  │
              │  └───────────────┘  │
              └─────────────────────┘
```

### Architecture Patterns

**Pattern** | **Usage** | **Benefit**
------------|-----------|------------
Layered Architecture | Data, Logic, UI separation | Maintainability, testability
MVC (Modified) | Streamlit=View, analysis.py=Controller, data=Model | Clear separation of concerns
Caching Pattern | `@st.cache_data` decorators | Performance optimization
Pipeline Pattern | Load → Clean → Analyze → Visualize | Clear data flow
Component-Based UI | Modular tab components | Reusability, isolation

---

## System Components

### 1. Entry Point Layer

**Component**: `dashboard.py`

**Responsibilities**:
- Parse CLI arguments
- Initialize Streamlit app
- Configure page layout
- Orchestrate data flow
- Manage global filters
- Route to tab components

**Key Functions**:
```python
main()              # CLI entry point
streamlit_app(df)   # Streamlit app initialization
```

**Data Flow**:
```
CLI Args → load_data() → clean_data() → apply_filters()
→ render_tabs() → display_output()
```

### 2. Data Layer

**Component**: `src/data_processing.py`

**Responsibilities**:
- File I/O (Excel/CSV)
- Schema validation
- Data type coercion
- Missing value handling
- Date normalization
- Text standardization
- Data caching

**Key Functions**:
```python
load_data(file_path_or_buffer) → DataFrame
clean_data(df) → DataFrame
```

**Caching Strategy**:
- Cache key: File path/buffer + nrows
- Invalidation: File content change (automatic)
- Expiration: Session-based (Streamlit default)

**Data Transformations**:
1. Load raw data (multi-sheet support)
2. Rename columns (fuzzy matching)
3. Coerce data types (numeric, dates)
4. Fill missing values (intelligent defaults)
5. Create synthetic columns (`date__ym`)
6. Remove invalid rows (e.g., missing dates)
7. Return cleaned DataFrame

### 3. Business Logic Layer

**Component**: `src/analysis.py`

**Responsibilities**:
- Compute KPIs
- Calculate growth metrics
- Perform RFM analysis
- Cluster customers
- Aggregate by dimensions

**Key Functions**:
```python
compute_top_level_kpis(df) → dict
compute_client_metrics(df) → DataFrame
compute_product_metrics(df) → DataFrame
compute_region_metrics(df) → DataFrame
compute_rfm_clusters(client_df) → DataFrame
```

**KPI Computation Pipeline**:
```
Input DataFrame
    ↓
Group by dimensions (client/product/region)
    ↓
Aggregate metrics (sum/count/nunique)
    ↓
Calculate derived metrics (growth %, RFM scores)
    ↓
Apply ML (K-Means clustering)
    ↓
Return results (dict/DataFrame)
```

### 4. Visualization Layer

**Component**: `src/charts.py`

**Responsibilities**:
- Generate Plotly charts
- Apply consistent styling
- Handle empty data states

**Key Functions**:
```python
fig_top_level(kpis) → Figure
fig_top_products(prod_df) → Figure
fig_region_map(region_df) → Figure
```

**Chart Types**:
- Line charts: Time-series trends
- Bar charts: Comparisons (grouped/stacked)
- Pie charts: Distributions
- Choropleth: Geographic data

### 5. UI Layer

**Component**: `src/ui_helpers.py`

**Responsibilities**:
- Custom CSS injection
- Reusable UI components
- Filter widgets

**Key Functions**:
```python
apply_custom_styles() → None
checkbox_filter(label, options, ...) → list
```

**Design System**:
- Gradient metric cards
- Professional tab styling
- Color palette management

### 6. Utility Layer

**Component**: `src/utils.py`

**Responsibilities**:
- Date filtering
- Growth calculations
- AI insights (optional)
- Report export

**Key Functions**:
```python
filter_by_date(df, years, months) → DataFrame
calculate_growth(current, previous) → float
ai_insights_summary(...) → str
export_reports(...) → None
```

### 7. Tab Components

**Components**: `src/tabs/*.py`

Each tab is a self-contained module with a single render function:

**Tab** | **Module** | **Function** | **Purpose**
--------|------------|--------------|------------
Executive Overview | `executive_overview.py` | `render_executive_overview()` | KPIs, revenue trends
Product Intelligence | `product_intelligence.py` | `render_product_intelligence()` | Product performance
Customer & Market | `customer_market.py` | `render_customer_market()` | Client segments, regions
Growth & Insights | `growth_insights.py` | `render_growth_insights()` | Spike detection, YoY
Vietnam Focus | `vietnam_focus.py` | `render_vietnam_focus()` | Vietnam-specific analysis
Product Launching | `product_launching.py` | `render_product_launching()` | Launch tracking, active customers

**Tab Rendering Pattern**:
```python
def render_<tab_name>(df_curr, df_prev, ...):
    # 1. Compute tab-specific metrics
    metrics = compute_metrics(df_curr)

    # 2. Display KPIs
    st.metric("Metric Name", value, delta)

    # 3. Render visualizations
    fig = px.bar(...)
    st.plotly_chart(fig)

    # 4. Display tables
    st.dataframe(data)
```

---

## Data Architecture

### Data Schema

**Core Columns** (12 required):
```
Year             int64      Sales year
Month            int64      Sales month (1-12)
Name of client   object     Customer name
Channel by Sales Person  object  Sales channel
Region           object     Geographic region
Country          object     Country name
Name of product  object     Product name
Kind of fruit    object     Fruit type
SKU              object     Product SKU
Type of product  object     Product category
Sold             float64    Revenue/Sales amount
Quantity (KG)    float64    Volume in KG
```

**Synthetic Columns** (added during cleaning):
```
date__ym         datetime64  Synthetic date (YYYY-MM-01)
```

**Derived Columns** (computed during analysis):
```
# Client metrics
recency_days     int64      Days since last order
frequency        int64      Total order count
monetary         float64    Total revenue
r_score          int64      Recency score (1-5)
f_score          int64      Frequency score (1-5)
m_score          int64      Monetary score (1-5)
rfm_score        int64      Composite RFM score
cluster          int64      K-Means cluster (0-3)

# Product metrics
price_per_kg     float64    Revenue / KG
contribution     float64    Revenue % contribution

# Growth metrics
Growth_Pct       float64    (Current - Baseline) / Baseline
```

### Data Flow Diagram

```
┌─────────────┐
│ Excel/CSV   │
│  Data File  │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ load_data()  │  Read Excel (all sheets) or CSV
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ clean_data() │  Validate, normalize, create date__ym
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Cleaned DataFrame│ (Cached)
└──────┬───────────┘
       │
       ├─────────────────────┬────────────────────┬──────────────┐
       ▼                     ▼                    ▼              ▼
┌──────────────┐  ┌────────────────┐  ┌───────────────┐  ┌─────────────┐
│ Apply Global │  │ Compute KPIs   │  │ Compute Client│  │ Compute Prod│
│   Filters    │  │ (top_level)    │  │   Metrics     │  │  Metrics    │
└──────┬───────┘  └────────┬───────┘  └───────┬───────┘  └──────┬──────┘
       │                   │                  │                  │
       │                   │                  │                  │
       ▼                   ▼                  ▼                  ▼
┌──────────────┐  ┌────────────────┐  ┌───────────────┐  ┌─────────────┐
│  df_curr     │  │  kpis (dict)   │  │ client_df     │  │  prod_df    │
│  df_prev     │  │                │  │ (with RFM)    │  │             │
└──────┬───────┘  └────────┬───────┘  └───────┬───────┘  └──────┬──────┘
       │                   │                  │                  │
       └───────────────────┴──────────────────┴──────────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │  Tab Renders  │
                   │ (Visualizations)
                   └───────────────┘
```

### Data Storage

**Current State** (File-Based):
- **Format**: Excel (.xlsx) or CSV
- **Location**: Local file system
- **Size**: < 100,000 rows (recommended)
- **Refresh**: Manual upload

**Future State** (Database):
- **Format**: PostgreSQL or MongoDB
- **Location**: Cloud database (AWS RDS, GCP Cloud SQL)
- **Size**: Unlimited (with indexing)
- **Refresh**: Real-time sync

---

## Application Architecture

### Execution Flow

```
1. User launches app
   └─ streamlit run dashboard.py

2. Streamlit server starts
   └─ Calls streamlit_app(df)

3. App initialization
   ├─ Set page config (layout='wide')
   ├─ Apply custom CSS
   └─ Show file uploader (if no CLI file)

4. Data loading
   ├─ load_data() [CACHED]
   └─ clean_data() [CACHED]

5. Sidebar filters
   ├─ Year (checkbox)
   ├─ Month (checkbox)
   ├─ Region (checkbox)
   ├─ Channel (checkbox, if exists)
   └─ Country (checkbox, if exists)

6. Data filtering
   ├─ filter_by_date(df, years, months) → df_curr
   └─ filter_by_date(df, [prev_year], months) → df_prev (if applicable)

7. Tab rendering
   ├─ Tab 1: Executive Overview
   ├─ Tab 2: Product Intelligence
   ├─ Tab 3: Customer & Market
   ├─ Tab 4: Growth & Insights
   ├─ Tab 5: Vietnam Focus
   └─ Tab 6: Product Launching

8. User interaction
   ├─ Change filters → Re-run steps 6-7
   ├─ Switch tabs → Render selected tab
   └─ Export data → save CSV/JSON
```

### State Management

**Streamlit Session State** (Future Enhancement):
```python
# Filter state
st.session_state.selected_years = [2024]
st.session_state.selected_months = [1, 2, 3, ...]

# Tab-specific state
st.session_state.selected_products = ['Product A', 'Product B']
st.session_state.spike_threshold = 0.3
```

**Current Approach**:
- Filters stored in local variables
- Re-computed on every widget interaction
- Cached data prevents redundant I/O

### Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                      dashboard.py                            │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Sidebar Filters (Global State)                    │    │
│  │  - Year, Month, Region, Channel, Country           │    │
│  └───────────────┬────────────────────────────────────┘    │
│                  │ Propagate filters                        │
│                  ▼                                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Data Filtering                                     │    │
│  │  df → df_curr (current period)                     │    │
│  │  df → df_prev (comparison period)                  │    │
│  └───────────────┬────────────────────────────────────┘    │
│                  │ Pass filtered data                       │
│                  ▼                                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Tab Rendering                                      │    │
│  │  - Each tab receives df_curr, df_prev, ...         │    │
│  │  - Computes tab-specific metrics                   │    │
│  │  - Renders UI independently                        │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## UI Architecture

### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Header: 🚀 Business Performance: 2024                      │
├─────────────────────────────────────────────────────────────┤
│  Tabs:  [Executive] [Product] [Customer] [Growth] ...      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┬────────────┬────────────┬────────────┐     │
│  │  Metric 1  │  Metric 2  │  Metric 3  │  Metric 4  │     │
│  │  (Revenue) │  (Volume)  │    (AOV)   │  (Clients) │     │
│  └────────────┴────────────┴────────────┴────────────┘     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Chart: Monthly Revenue Trend                        │  │
│  │  [Interactive Plotly visualization]                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Table: Top Products                                 │  │
│  │  [Sortable, filterable dataframe]                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌───────────────────┐
│  Sidebar          │
│  ┌─────────────┐  │
│  │ Control     │  │
│  │ Panel       │  │
│  ├─────────────┤  │
│  │ ▼ Years     │  │
│  │ ☑ 2023      │  │
│  │ ☑ 2024      │  │
│  │ ☐ 2025      │  │
│  ├─────────────┤  │
│  │ ▼ Months    │  │
│  │ ☑ Jan       │  │
│  │ ☑ Feb       │  │
│  │ ...         │  │
│  ├─────────────┤  │
│  │ ▼ Regions   │  │
│  │ ☑ South     │  │
│  │ ☑ North     │  │
│  │ ...         │  │
│  └─────────────┘  │
└───────────────────┘
```

### Component Hierarchy

```
App
├── Sidebar
│   ├── Control Panel
│   │   ├── Year Filter (Expander)
│   │   ├── Month Filter (Expander)
│   │   ├── Region Filter (Expander)
│   │   ├── Channel Filter (Expander)
│   │   └── Country Filter (Expander)
│   └── Debug Info (Expander)
└── Main Content
    ├── Header (Title)
    ├── Tabs (Container)
    │   ├── Tab 1: Executive Overview
    │   │   ├── KPI Cards (Columns)
    │   │   ├── Revenue Trend Chart
    │   │   └── Chart Type Toggle (Radio)
    │   ├── Tab 2: Product Intelligence
    │   │   ├── Product Table (DataFrame)
    │   │   ├── Product Comparison (Multiselect)
    │   │   └── Comparison Chart
    │   ├── Tab 3: Customer & Market
    │   │   ├── Regional Performance (Chart)
    │   │   ├── Client Segments (Pie Chart)
    │   │   └── Top Clients Table
    │   ├── Tab 4: Growth & Insights
    │   │   ├── Product Type Analysis
    │   │   ├── Spike Detection (Selectbox)
    │   │   └── YoY Growth Drivers (Waterfall)
    │   ├── Tab 5: Vietnam Focus
    │   │   ├── Category Focus (Selectbox)
    │   │   ├── Top 10 Table
    │   │   └── Regional Breakdown Chart
    │   └── Tab 6: Product Launching
    │       ├── Filter Form (Multiselect)
    │       ├── Launch Table (DataFrame)
    │       └── Active Customers (Dialog)
    └── Footer (Optional)
```

### Responsive Design

**Breakpoints**:
- Desktop: > 1024px (optimized)
- Tablet: 768px - 1024px (supported)
- Mobile: < 768px (limited support)

**Layout Strategy**:
- Use `st.columns()` for responsive grids
- Charts auto-scale with `use_container_width=True`
- Tables use horizontal scroll on small screens

---

## Deployment Architecture

### Local Development

```
Developer Machine
├── Python 3.13+ (venv)
├── requirements.txt installed
└── data.xlsx in project root

Launch:
$ source venv/bin/activate
$ streamlit run dashboard.py
```

### Streamlit Cloud (Recommended)

```
GitHub Repository
    │
    └─ Detected by Streamlit Cloud
       │
       ├─ Automatic deployment
       ├─ requirements.txt parsed
       └─ Secrets management (API keys)

Access:
https://<app-name>.streamlit.app
```

**Configuration** (`.streamlit/config.toml`):
```toml
[server]
maxUploadSize = 200  # MB

[theme]
primaryColor = "#1E90FF"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F4FF"
```

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]
```

**Docker Compose** (`docker-compose.yml`):
```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

### Cloud Platform Deployment

**AWS**:
- EC2: Run Docker container
- ECS: Managed container service
- Elastic Beanstalk: Streamlit app hosting

**GCP**:
- Cloud Run: Serverless containers
- Compute Engine: VM-based hosting
- App Engine: Managed platform

**Azure**:
- App Service: Web app hosting
- Container Instances: Managed containers

---

## Security Architecture

### Data Security

**At Rest**:
- Data files excluded from Git (`.gitignore`)
- No sensitive data in repository

**In Transit**:
- HTTPS for Streamlit Cloud (automatic)
- Self-signed SSL for local (optional)

**Access Control** (Future):
- User authentication (OAuth, SAML)
- Role-based access (admin, viewer)
- Row-level security (filter by region/team)

### Application Security

**Input Validation**:
```python
# File upload validation
if uploaded_file.type not in ['application/vnd.ms-excel', 'text/csv']:
    st.error("Invalid file type")

# Numeric input validation
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
```

**API Key Management**:
```python
# Environment variable (not hardcoded)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Streamlit secrets (production)
openai_api_key = st.secrets["openai"]["api_key"]
```

**No SQL Injection** (Future Database):
- Use parameterized queries
- ORM layer (SQLAlchemy)

---

## Performance Considerations

### Caching Strategy

```python
@st.cache_data(show_spinner=False)
def load_data(file_path):
    # Cached for same file_path
    ...

@st.cache_data(show_spinner=False)
def clean_data(df):
    # Cached for same DataFrame hash
    ...
```

**Cache Invalidation**:
- File change: Automatic (Streamlit hash-based)
- Manual: Clear cache button in UI

### Data Processing Optimization

1. **Vectorized Operations**: Use Pandas/NumPy native functions
2. **Early Filtering**: Reduce dataset before aggregation
3. **Chunking** (Future): Process large files in chunks

### Chart Rendering Optimization

1. **Limit Data Points**: Display top N rows in tables
2. **Downsample**: For time-series with 1000+ points
3. **Lazy Loading**: Render charts only when tab is active (built-in)

### Memory Management

- **Monitor**: Use `df.memory_usage()` for profiling
- **Release**: Delete temporary DataFrames
- **Garbage Collection**: Automatic Python GC

---

## Scalability & Future State

### Current Limitations

- **Single User**: No concurrent user support
- **File-Based**: Manual data refresh
- **Memory-Bound**: Limited to ~100k rows
- **No History**: No audit trail or versioning

### Scaling Strategy

**Phase 1**: Current (File-Based)
```
User → Streamlit → DataFrame → Visualizations
```

**Phase 2**: Database Integration
```
User → Streamlit → PostgreSQL → DataFrames → Visualizations
                        ↑
                   ETL Pipeline
```

**Phase 3**: Microservices
```
                 ┌─ Analytics Service
User → API Gateway ─┤
                 ├─ Data Service
                 └─ Reporting Service
```

### Future Architecture Components

**1. Backend API** (FastAPI/Flask):
```python
@app.get("/api/kpis")
def get_kpis(year: int, region: str):
    # Compute KPIs
    return {"revenue": 1000000, ...}
```

**2. Real-Time Data Sync**:
- CDC (Change Data Capture) from source systems
- Kafka for event streaming
- Incremental updates

**3. Advanced Analytics**:
- ML models for forecasting
- Anomaly detection (Isolation Forest)
- Recommendation engine

**4. Multi-Tenancy**:
- User authentication (Auth0, Okta)
- Row-level security
- Tenant isolation

---

## Diagrams

### Deployment Diagram

```
┌────────────────────────────────────────────────────────────┐
│                     Internet                                │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
               ┌────────────────┐
               │  Load Balancer │
               │   (AWS ALB)    │
               └────────┬───────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────┐               ┌───────────────┐
│  ECS Task 1   │               │  ECS Task 2   │
│ (Streamlit)   │               │ (Streamlit)   │
└───────┬───────┘               └───────┬───────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
                ┌───────────────┐
                │   RDS Postgres│
                │   (Database)  │
                └───────────────┘
```

### Sequence Diagram (Data Load)

```
User       Dashboard    load_data    clean_data    Streamlit Cache
 │             │            │             │              │
 │─Upload─────>│            │             │              │
 │             │─Call──────>│             │              │
 │             │            │─Check Cache─────────────────>│
 │             │            │<───────Cache Hit─────────────│
 │             │            │─Return DataFrame─>│          │
 │             │<───────────────────────────────│          │
 │             │─Call──────────────────────────>│          │
 │             │            │             │─Check Cache────>│
 │             │            │             │<────Cache Hit───│
 │             │            │             │─Return──────────>│
 │             │<───────────────────────────────────────────│
 │<─Display───│            │             │              │
```

---

## Conclusion

The Professional Sales Analytics Dashboard follows a **layered architecture** with clear separation between data processing, business logic, and presentation layers. This design ensures:

- **Maintainability**: Modular components are easy to update
- **Scalability**: Architecture supports future database integration
- **Performance**: Caching and optimization strategies minimize latency
- **Security**: Input validation and secrets management protect data
- **Extensibility**: New tabs, metrics, and features can be added independently

The current file-based architecture is suitable for **small to medium datasets** (< 100k rows) with **manual refresh cycles**. For production deployment at scale, migration to a database-backed architecture with real-time sync is recommended.

---

**Maintained By**: Development Team
**Last Reviewed**: 2025-12-08
**Next Review**: Q1 2026
