# Professional Sales Analytics Dashboard

A comprehensive business intelligence platform for sales performance analysis, customer segmentation, and market insights built with Python and Streamlit.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The Professional Sales Analytics Dashboard transforms sales data into actionable insights through:

- **Real-time KPI tracking**: Revenue, volume, AOV, active clients with YoY/MoM growth
- **Customer intelligence**: RFM-based segmentation, churn prediction, gap analysis
- **Product analytics**: Launch tracking, performance comparison, spike detection
- **Regional insights**: Geographic performance with Vietnam-specific focus
- **Interactive visualizations**: Professional Plotly charts with custom styling

**Technology Stack**: Python 3.13+ | Streamlit | Pandas | Plotly | scikit-learn

---

## Features

### Executive Overview
- Key performance indicators with growth indicators
- Monthly revenue trends (YoY/MoM comparison)
- Interactive chart types (bar/line toggle)

### Product Intelligence
- Product performance table with contribution percentages
- Multi-product comparison tool
- Performance analysis by product type

### Customer & Market Analysis
- Revenue breakdown by country
- RFM-based customer segmentation (Diamond/Gold/Silver/Bronze)
- Top clients with 6-month fruit variety gap analysis

### Growth & Insights
- Product type performance analysis
- Automated spike detection (>30% growth)
- YoY growth drivers (new/lost/existing clients)
- Client-level drill-down for growth investigation

### Vietnam Focus
- Category-specific performance tracking
- Regional breakdown (South/North/Center)
- Top 10 products with YoY comparison

### Product Launching
- New product identification
- Active customer tracking (≥2 orders in 6 months)
- Customer journey analysis

---

## Quick Start

### Prerequisites
- Python 3.13 or higher
- pip package manager
- Excel/CSV file with sales data

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd nhan

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard.py

# Or run with a specific file
streamlit run dashboard.py -- --file data.xlsx --mode streamlit
```

The dashboard will open in your default browser at `http://localhost:8501`

---

## Usage

### Data Format

Your Excel/CSV file should include these columns:

```
Year, Month, Name of client, Channel by Sales Person,
Region, Country, Name of product, Kind of fruit, SKU,
Type of product, Sold, Quantity (KG)
```

**Example**:
```csv
Year,Month,Name of client,Channel by Sales Person,Region,Country,Name of product,Kind of fruit,SKU,Type of product,Sold,Quantity (KG)
2024,1,Client A,Retail,South,Viet Nam,Apple Puree,Apple,A01,FROZEN PUREE,100,100
```

### CLI Mode (Batch KPI Generation)

```bash
# Generate KPI reports to CSV/JSON
python dashboard.py --file sales.xlsx --mode kpis --out reports_out

# View summary
cat reports_out/kpis.json
```

### Streamlit Mode (Interactive Dashboard)

1. **Upload Data**: Use file uploader or pass via CLI
2. **Apply Filters**: Select years, months, regions in sidebar
3. **Explore Tabs**: Navigate through 6 interactive analysis tabs
4. **Export Results**: Download filtered data or reports

---

## Project Structure

```
nhan/
├── src/                        # Core application modules
│   ├── tabs/                   # Dashboard tab modules
│   │   ├── customer_market.py  # Customer & market analysis
│   │   ├── executive_overview.py # Executive KPI dashboard
│   │   ├── growth_insights.py  # Growth & spike detection
│   │   ├── product_intelligence.py # Product performance
│   │   ├── product_launching.py # Product launch tracking
│   │   └── vietnam_focus.py    # Vietnam market focus
│   ├── analysis.py             # KPI computation & metrics
│   ├── charts.py               # Plotly chart helpers
│   ├── data_processing.py      # Data loading & cleaning
│   ├── ui_helpers.py           # UI components & styling
│   └── utils.py                # Utility functions
├── docs/                       # Documentation
│   ├── codebase-summary.md     # Detailed codebase overview
│   ├── project-overview-pdr.md # Product requirements
│   ├── code-standards.md       # Coding standards
│   └── system-architecture.md  # System architecture
├── dashboard.py                # Main application entry
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## Documentation

Comprehensive documentation is available in the `/docs` directory:

- **[Codebase Summary](docs/codebase-summary.md)**: Detailed overview of all modules, functions, and algorithms
- **[Project Overview & PDR](docs/project-overview-pdr.md)**: Product requirements, user stories, and success criteria
- **[Code Standards](docs/code-standards.md)**: Coding conventions, best practices, and architectural patterns
- **[System Architecture](docs/system-architecture.md)**: High-level architecture, data flow, and deployment strategies

---

## Development

### Setup Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (future)
pytest tests/

# Check code style (future)
flake8 src/
```

### Code Standards

- Follow PEP 8 style guide
- Maximum line length: 120 characters
- Use snake_case for variables/functions
- Add docstrings for all functions
- See [Code Standards](docs/code-standards.md) for details

### Adding a New Tab

1. Create new file in `src/tabs/`: `my_new_tab.py`
2. Implement render function: `render_my_new_tab(df_curr, df_prev, ...)`
3. Import in `dashboard.py`: `from src.tabs.my_new_tab import render_my_new_tab`
4. Add tab in UI: `tab7 = st.tabs(["...", "My New Tab"])`
5. Call render function: `with tab7: render_my_new_tab(df_curr, df_prev, ...)`

### Data Processing Pipeline

```
Excel/CSV → load_data() → clean_data() → apply_filters()
→ compute_metrics() → render_visualizations() → Streamlit UI
```

---

## Deployment

### Local Development

```bash
streamlit run dashboard.py
```

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Configure secrets (API keys) in dashboard settings
4. Deploy automatically on push

### Docker

```bash
# Build image
docker build -t sales-dashboard .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data sales-dashboard
```

### Cloud Platforms

- **AWS**: EC2, ECS, Elastic Beanstalk
- **GCP**: Cloud Run, Compute Engine, App Engine
- **Azure**: App Service, Container Instances

See [System Architecture](docs/system-architecture.md) for detailed deployment strategies.

---

## Configuration

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 200  # MB

[theme]
primaryColor = "#1E90FF"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F4FF"
```

### Environment Variables

```bash
# Optional: OpenAI API key for AI insights
export OPENAI_API_KEY="sk-..."

# Optional: Default data file path
export DATA_FILE_PATH="./data.xlsx"
```

---

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **openpyxl**: Excel file support
- **plotly**: Interactive visualizations
- **streamlit**: Web dashboard framework
- **scikit-learn**: Machine learning (K-Means clustering)
- **numpy**: Numerical operations

### Optional
- **openai**: AI-powered insights generation

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: data.xlsx not found`
- **Solution**: Ensure data file is in project root or use file uploader in Streamlit

**Issue**: `ModuleNotFoundError: No module named 'streamlit'`
- **Solution**: Activate virtual environment and run `pip install -r requirements.txt`

**Issue**: Dashboard loads slowly
- **Solution**: Reduce dataset size or filter data before uploading

**Issue**: Charts not displaying
- **Solution**: Clear browser cache and refresh page

### Debug Mode

Enable debug info in sidebar to inspect:
- Row count and columns
- Unique years and regions
- Sample data preview

---

## Performance Tips

- **Data Size**: Recommended < 100,000 rows for optimal performance
- **Filtering**: Use sidebar filters to reduce dataset before analysis
- **Caching**: Data is cached automatically - refresh only when needed
- **Export**: Use CLI mode for batch processing large datasets

---

## Roadmap

### Phase 1 (Completed)
- ✅ Data ingestion and cleaning
- ✅ Six interactive dashboard tabs
- ✅ Global filtering and YoY/MoM comparison
- ✅ Export functionality (CSV/JSON)

### Phase 2 (Planned - Q1 2026)
- Time-series forecasting
- Advanced anomaly detection
- Scheduled email reports
- PDF export for executive summaries

### Phase 3 (Planned - Q2 2026)
- Database integration (PostgreSQL)
- Real-time data sync
- API endpoints
- Multi-source data fusion

### Phase 4 (Planned - Q3 2026)
- User authentication and roles
- Collaborative features
- Dashboard templates

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow [Code Standards](docs/code-standards.md)
4. Commit changes (`git commit -m 'feat: Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Commit Message Format

```
<type>(<scope>): <subject>

feat: New feature
fix: Bug fix
docs: Documentation changes
style: Code style changes
refactor: Code refactoring
test: Test updates
```

---

## Support

For questions, issues, or feature requests:

- **Documentation**: Check `/docs` folder
- **Issues**: Open a GitHub issue
- **Email**: [Your contact email]

---

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Charts powered by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)

---

## License

[Specify your license here - e.g., MIT, Apache 2.0]

---

## Change Log

### Version 1.0 (2025-12-08)
- Initial release
- Six interactive dashboard tabs
- YoY/MoM comparison
- RFM customer segmentation
- Spike detection algorithm
- Vietnam market focus
- Product launch tracking

---

**Last Updated**: 2025-12-08
**Version**: 1.0
**Maintained By**: Development Team
