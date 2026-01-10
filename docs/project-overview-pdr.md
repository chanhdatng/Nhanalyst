# Project Overview & Product Development Requirements (PDR)

**Project Name**: Professional Sales Analytics Dashboard
**Version**: 1.0
**Last Updated**: 2025-12-08
**Status**: Production-Ready

---

## Executive Summary

The Professional Sales Analytics Dashboard is a comprehensive business intelligence platform designed to transform sales data into actionable insights. Built with Python and Streamlit, this system provides real-time analytics for sales performance, customer segmentation, product intelligence, and regional market analysis with specialized focus on Vietnam operations.

### Vision Statement

Empower sales teams and business leaders with data-driven insights to optimize revenue growth, improve customer retention, and make informed strategic decisions.

### Key Stakeholders

- **Sales Teams**: Regional sales managers tracking performance metrics
- **Executive Leadership**: C-level executives monitoring business health
- **Product Managers**: Teams analyzing product launches and market fit
- **Marketing Teams**: Customer segmentation and targeting insights
- **Data Analysts**: Deep-dive analysis and custom reporting

---

## Product Objectives

### Primary Objectives

1. **Revenue Visibility**: Provide real-time visibility into sales performance across dimensions (region, product, customer, channel)
2. **Growth Analysis**: Enable YoY/MoM growth tracking with automated spike detection
3. **Customer Intelligence**: Deliver RFM-based customer segmentation and churn prediction
4. **Product Insights**: Track product performance, launches, and market penetration
5. **Regional Focus**: Provide Vietnam-specific analysis with regional breakdown

### Success Metrics

- **User Adoption**: 80%+ of sales managers using dashboard weekly
- **Data Freshness**: < 24 hour lag from sales transaction to dashboard
- **Insight Actionability**: 50%+ of growth spikes result in sales actions
- **Performance**: < 5 second load time for standard reports
- **Data Quality**: 95%+ data accuracy and completeness

---

## Functional Requirements

### FR-1: Data Ingestion & Processing

**Priority**: P0 (Critical)

**Description**: System must ingest, validate, and clean sales data from Excel/CSV files

**Requirements**:
- FR-1.1: Support Excel (.xlsx, .xls) and CSV file formats
- FR-1.2: Handle multi-sheet Excel files with automatic concatenation
- FR-1.3: Validate against expected schema (12 core columns)
- FR-1.4: Perform fuzzy column matching for header variations
- FR-1.5: Normalize text fields (strip, title-case, standardize)
- FR-1.6: Create synthetic date column from Year/Month fields
- FR-1.7: Handle missing values with intelligent defaults
- FR-1.8: Cache processed data for performance

**Expected Schema**:
```
Year, Month, Name of client, Channel by Sales Person,
Region, Country, Name of product, Kind of fruit, SKU,
Type of product, Sold, Quantity (KG)
```

**Acceptance Criteria**:
- ✓ Load 10,000+ row files in < 10 seconds
- ✓ Handle 100% of column name variations (common cases)
- ✓ 0% data loss during cleaning
- ✓ Clear error messages for invalid files

### FR-2: KPI Computation

**Priority**: P0 (Critical)

**Description**: Calculate core business metrics and KPIs

**Requirements**:
- FR-2.1: Total revenue, total volume (KG), active clients
- FR-2.2: YoY growth (Year-over-Year comparison)
- FR-2.3: MoM growth (Month-over-Month comparison)
- FR-2.4: Average Order Value (AOV)
- FR-2.5: New client count (last 12 months)
- FR-2.6: Churned client count (no orders in last 3 months)
- FR-2.7: Top products and fruit types by revenue

**Acceptance Criteria**:
- ✓ All KPIs computed correctly vs manual Excel calculations
- ✓ Handle edge cases (division by zero, missing periods)
- ✓ Display growth as percentage with proper formatting

### FR-3: Customer Segmentation

**Priority**: P1 (High)

**Description**: Segment customers using RFM analysis and clustering

**Requirements**:
- FR-3.1: Calculate RFM scores (Recency, Frequency, Monetary)
- FR-3.2: Apply K-Means clustering (4 clusters)
- FR-3.3: Segment into Diamond/Gold/Silver/Bronze tiers
- FR-3.4: Display top clients with 6-month gap analysis
- FR-3.5: Show fruit variety gaps per client

**Segmentation Thresholds**:
- Diamond: > 50,000 KG
- Gold: > 10,000 KG
- Silver: > 1,000 KG
- Bronze: ≤ 1,000 KG

**Acceptance Criteria**:
- ✓ RFM scores match industry standard calculations
- ✓ Clustering is stable across runs (fixed random seed)
- ✓ Gap analysis identifies upsell opportunities

### FR-4: Product Intelligence

**Priority**: P1 (High)

**Description**: Track product performance and launches

**Requirements**:
- FR-4.1: Product performance table with contribution %
- FR-4.2: Product comparison tool (multi-select)
- FR-4.3: Performance by Type of Product analysis
- FR-4.4: Identify new product launches (YoY comparison)
- FR-4.5: Track active customers per product (≥ 2 orders in 6m)
- FR-4.6: Display customer journey for launched products

**Acceptance Criteria**:
- ✓ Contribution % sums to 100%
- ✓ Comparison charts display correctly for 2-10 products
- ✓ Active customer count matches manual validation

### FR-5: Growth & Spike Detection

**Priority**: P1 (High)

**Description**: Identify unusual growth patterns and spikes

**Requirements**:
- FR-5.1: Detect SKU spikes with > 30% growth threshold
- FR-5.2: Support YoY and MoM comparison modes
- FR-5.3: Filter by minimum volume threshold
- FR-5.4: Drill down to customers driving spike
- FR-5.5: Display growth drivers (new/lost/existing clients)
- FR-5.6: Show YoY growth waterfall charts

**Spike Algorithm**:
```
Growth % = (Current - Baseline) / Baseline
Spike = Growth > 30% AND Current Volume >= Min Threshold
```

**Acceptance Criteria**:
- ✓ Detect 100% of spikes above threshold
- ✓ Handle new products (baseline = 0) correctly
- ✓ Display client drill-down within 2 clicks

### FR-6: Regional Analysis

**Priority**: P1 (High)

**Description**: Analyze sales by geographic region

**Requirements**:
- FR-6.1: Revenue by Country (horizontal bar chart)
- FR-6.2: Vietnam-specific analysis with category focus
- FR-6.3: Regional breakdown (South/North/Center)
- FR-6.4: Top 10 products per category in Vietnam
- FR-6.5: YoY comparison for Vietnam products

**Acceptance Criteria**:
- ✓ Choropleth map displays correctly (or fallback to bar)
- ✓ Vietnam filter case-insensitive ('Viet Nam', 'Vietnam')
- ✓ Regional colors consistent (South=Green, North=Blue, Center=Orange)

### FR-7: Interactive Filtering

**Priority**: P0 (Critical)

**Description**: Global filters to slice data by dimensions

**Requirements**:
- FR-7.1: Year filter (multi-select, default: current year)
- FR-7.2: Month filter (multi-select, default: all)
- FR-7.3: Region filter (multi-select, default: all)
- FR-7.4: Channel filter (multi-select if column exists)
- FR-7.5: Country filter (multi-select if column exists)
- FR-7.6: Filters apply to all tabs simultaneously

**Acceptance Criteria**:
- ✓ Filters update all tabs without page reload
- ✓ Default selections load < 3 seconds
- ✓ Handle "no results" state gracefully

### FR-8: Visualization & Charts

**Priority**: P1 (High)

**Description**: Interactive, professional visualizations

**Requirements**:
- FR-8.1: Line charts for time-series trends
- FR-8.2: Bar charts for comparisons (grouped/stacked)
- FR-8.3: Pie/donut charts for distributions
- FR-8.4: Waterfall charts for growth attribution
- FR-8.5: Choropleth maps for geographic data
- FR-8.6: Consistent color scheme and branding
- FR-8.7: Responsive design (desktop/tablet/mobile)

**Color Palette**:
- Primary: #1E90FF (Dodger Blue)
- Secondary: #D3D3D3 (Light Gray)
- Gradient: #2E3192 → #1BFFFF (Dark Blue → Cyan)
- Positive: #2ca02c (Green)
- Negative: #f8d7da (Red)

**Acceptance Criteria**:
- ✓ All charts render in < 2 seconds
- ✓ Charts are interactive (hover, zoom, pan)
- ✓ Tooltips display relevant data

### FR-9: Export & Reporting

**Priority**: P2 (Medium)

**Description**: Export data and generate reports

**Requirements**:
- FR-9.1: Export KPIs to JSON format
- FR-9.2: Export tables to CSV format
- FR-9.3: CLI mode for batch KPI generation
- FR-9.4: Save monthly time-series data

**CLI Usage**:
```bash
python dashboard.py --file sales.xlsx --mode kpis --out reports_out
```

**Acceptance Criteria**:
- ✓ Exported files are valid JSON/CSV
- ✓ CLI mode runs without GUI dependencies
- ✓ File naming includes timestamp

### FR-10: AI Insights (Optional)

**Priority**: P3 (Low)

**Description**: Generate AI-powered insights using OpenAI

**Requirements**:
- FR-10.1: Integration with OpenAI API
- FR-10.2: Generate 5 action-oriented insights
- FR-10.3: Identify 3 risk areas
- FR-10.4: Graceful fallback if API unavailable

**Acceptance Criteria**:
- ✓ Insights are contextually relevant
- ✓ Fallback displays template insights
- ✓ API errors handled gracefully

---

## Non-Functional Requirements

### NFR-1: Performance

**Requirements**:
- NFR-1.1: Dashboard loads in < 5 seconds with 10,000 rows
- NFR-1.2: Filter updates complete in < 2 seconds
- NFR-1.3: Chart rendering in < 2 seconds
- NFR-1.4: Support datasets up to 100,000 rows
- NFR-1.5: Memory usage < 2GB for typical datasets

**Optimization Strategies**:
- Data caching with `@st.cache_data`
- Lazy tab rendering
- Pre-aggregated metrics
- Efficient Pandas operations

### NFR-2: Usability

**Requirements**:
- NFR-2.1: Intuitive navigation with clear tab labels
- NFR-2.2: Professional, modern UI design
- NFR-2.3: Consistent terminology across interface
- NFR-2.4: Helpful tooltips and info boxes
- NFR-2.5: Mobile-responsive layout
- NFR-2.6: Accessible color contrast (WCAG AA)

### NFR-3: Reliability

**Requirements**:
- NFR-3.1: Graceful error handling for invalid data
- NFR-3.2: Clear error messages with resolution steps
- NFR-3.3: Data validation before processing
- NFR-3.4: 99% uptime for production deployment
- NFR-3.5: Automatic recovery from transient errors

### NFR-4: Maintainability

**Requirements**:
- NFR-4.1: Modular architecture with clear separation
- NFR-4.2: Consistent code style (PEP 8)
- NFR-4.3: Comprehensive documentation
- NFR-4.4: Version control with Git
- NFR-4.5: Automated testing (unit, integration)

### NFR-5: Security

**Requirements**:
- NFR-5.1: No hardcoded credentials
- NFR-5.2: API keys passed via environment variables
- NFR-5.3: Input validation to prevent injection
- NFR-5.4: Data files excluded from version control
- NFR-5.5: Secure file upload validation

### NFR-6: Scalability

**Requirements**:
- NFR-6.1: Horizontal scaling for Streamlit deployment
- NFR-6.2: Database integration ready (future)
- NFR-6.3: Multi-tenant architecture support (future)
- NFR-6.4: Stateless session management

---

## User Stories

### Sales Manager Persona

**Story 1**: Track Regional Performance
> As a regional sales manager, I want to view revenue trends by country so that I can identify high-performing markets and allocate resources accordingly.

**Acceptance Criteria**:
- View revenue by country in bar chart
- Filter by specific time periods
- Compare YoY performance

**Story 2**: Identify Top Customers
> As a sales manager, I want to see my top 20 customers with their purchase patterns so that I can prioritize retention efforts.

**Acceptance Criteria**:
- Display top customers ranked by volume
- Show last order date and order frequency
- Highlight fruit variety gaps for upselling

### Executive Persona

**Story 3**: Monitor Business Health
> As a CEO, I want to see key KPIs (revenue, volume, clients, AOV) with growth indicators so that I can assess overall business performance at a glance.

**Acceptance Criteria**:
- KPI cards with current values
- YoY/MoM growth percentages
- Visual trend indicators (up/down arrows)

**Story 4**: Identify Growth Opportunities
> As a COO, I want to detect products with unusual growth so that I can investigate drivers and replicate success.

**Acceptance Criteria**:
- Automated spike detection (> 30% growth)
- Drill down to customers driving spike
- Export spike data for further analysis

### Product Manager Persona

**Story 5**: Track Product Launches
> As a product manager, I want to identify which products were launched this year and track their adoption so that I can measure launch success.

**Acceptance Criteria**:
- Identify new products (not in previous year)
- Show active customer count per product
- Display customer journey and retention

**Story 6**: Analyze Product Types
> As a product manager, I want to compare performance across product types (Frozen Puree, Frozen Fruit) so that I can optimize the product mix.

**Acceptance Criteria**:
- Group products by Type
- Show revenue and volume trends
- Compare YoY growth by type

---

## Technical Architecture Summary

### Technology Stack

- **Backend**: Python 3.13+
- **Web Framework**: Streamlit 1.x
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **Machine Learning**: scikit-learn (K-Means)
- **Optional**: OpenAI API (AI insights)

### Data Flow

```
Excel/CSV → load_data() → clean_data() → apply_filters()
→ compute_metrics() → render_visualizations() → Streamlit UI
```

### Deployment Options

1. **Local**: `streamlit run dashboard.py`
2. **Streamlit Cloud**: GitHub integration
3. **Docker**: Containerized deployment
4. **Cloud Platforms**: AWS/GCP/Azure

---

## Constraints & Assumptions

### Constraints

1. **Data Format**: Assumes consistent Excel/CSV schema
2. **Single File**: One data file per dashboard session
3. **No Real-Time**: Requires manual data refresh
4. **Limited History**: Performance degrades > 100k rows
5. **Desktop-First**: Optimized for desktop browsers

### Assumptions

1. **Data Quality**: Assumes 95%+ data completeness
2. **User Skill**: Assumes basic Excel/dashboard literacy
3. **Update Frequency**: Assumes weekly/monthly data updates
4. **Language**: English-only interface
5. **Timezone**: All dates assumed in single timezone

---

## Risks & Mitigation

### Risk 1: Data Quality Issues

**Impact**: High | **Probability**: Medium

**Mitigation**:
- Robust data validation and cleaning
- Clear error messages for invalid data
- Data debug panel in UI
- Documentation of expected schema

### Risk 2: Performance Degradation

**Impact**: Medium | **Probability**: Medium

**Mitigation**:
- Data caching with Streamlit
- Lazy loading of tabs
- Row sampling for testing (nrows parameter)
- Database integration for large datasets (future)

### Risk 3: User Adoption

**Impact**: High | **Probability**: Low

**Mitigation**:
- Intuitive UI design
- Comprehensive user documentation
- Training sessions for stakeholders
- Regular feature updates based on feedback

### Risk 4: Dependency Changes

**Impact**: Low | **Probability**: Medium

**Mitigation**:
- Pin dependency versions in requirements.txt
- Regular dependency updates and testing
- Containerized deployment (Docker)

---

## Roadmap & Future Enhancements

### Phase 1: MVP (Completed)
- ✓ Data ingestion and cleaning
- ✓ Core KPI computation
- ✓ Six interactive tabs
- ✓ Global filtering
- ✓ Export functionality

### Phase 2: Advanced Analytics (Q1 2026)
- Time-series forecasting (Prophet/ARIMA)
- Advanced anomaly detection (Isolation Forest)
- Custom RFM threshold configuration
- Scheduled email reports
- PDF export for executive summaries

### Phase 3: Data Integration (Q2 2026)
- Database integration (PostgreSQL/MongoDB)
- Real-time data sync
- API endpoints for external systems
- Multi-file/multi-source data fusion

### Phase 4: Collaboration (Q3 2026)
- User authentication and roles
- Shared dashboards and annotations
- Commenting on insights
- Dashboard templates and presets

### Phase 5: AI & Automation (Q4 2026)
- Advanced AI insights (GPT-4)
- Natural language queries
- Automated action recommendations
- Predictive churn modeling

---

## Compliance & Governance

### Data Privacy
- No PII (Personally Identifiable Information) stored
- Client names anonymized in shared reports
- GDPR-compliant data handling (EU markets)

### Version Control
- Git-based version control
- Semantic versioning (MAJOR.MINOR.PATCH)
- Change log maintenance

### Documentation Standards
- Code comments for complex logic
- Docstrings for all functions
- README for setup instructions
- User guide for end-users

---

## Success Criteria

The project is considered successful when:

1. ✓ All P0/P1 functional requirements are implemented
2. ✓ Dashboard loads < 5 seconds with standard datasets
3. ✓ 80%+ user adoption within sales teams
4. ✓ Zero critical bugs in production
5. ✓ Positive feedback from stakeholders (> 4/5 satisfaction)
6. ✓ Measurable impact on sales decisions (tracked via usage logs)

---

## Appendices

### Appendix A: Glossary

- **AOV (Average Order Value)**: Total revenue / Total orders
- **RFM**: Recency, Frequency, Monetary (customer segmentation model)
- **YoY**: Year-over-Year (comparison with same period last year)
- **MoM**: Month-over-Month (comparison with previous month)
- **SKU**: Stock Keeping Unit (unique product identifier)
- **Churn**: Customers with no orders in last 3 months

### Appendix B: Sample Data Schema

See FR-1 for detailed schema definition.

### Appendix C: Contact & Support

- **Project Owner**: Sales Operations Team
- **Technical Lead**: Data Analytics Team
- **Support**: [Internal Support Portal]
- **Documentation**: /docs folder in repository

---

**Document Version**: 1.0
**Approved By**: [Pending Stakeholder Review]
**Next Review Date**: Q1 2026
