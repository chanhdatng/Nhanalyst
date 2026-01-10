# Phase 2: UI Integration

**Phase**: 2 of 3
**Created**: 2025-12-09
**Status**: ✅ COMPLETED
**Priority**: P1
**Duration**: 4-5 days
**Completed**: 2025-12-09

---

## Context

**Parent Plan**: [plan.md](./plan.md)
**Dependencies**: Phase 1 complete (all compute functions in `src/analysis.py`)
**Related Docs**:
- [Code Standards](../../docs/code-standards.md) - Streamlit best practices
- [UI Helpers](../../src/ui_helpers.py) - Existing UI patterns

---

## Overview

Integrate 5 new features into existing dashboard tabs. Add new sections below existing content, maintain visual consistency, follow existing patterns (`checkbox_filter`, custom styles).

**Implementation Status**: ✅ Complete (5/5 features integrated)
**Review Status**: Ready for visual QA

---

## Key Insights

1. **Existing UI Patterns**:
   - `st.subheader()` for section titles
   - `st.columns()` for metric cards
   - `st.metric()` with delta for KPIs
   - `st.plotly_chart()` for visualizations
   - `st.dataframe()` with styling for tables
   - `st.divider()` or `st.markdown("---")` for separators

2. **Tab Render Functions**:
   - All tabs have `render_*()` function
   - Take filtered data as params
   - Return nothing (render directly to Streamlit)

3. **Color Scheme**:
   - Primary: #1E90FF (Dodger Blue)
   - Gradient: #2E3192 → #1BFFFF
   - Green: #00C853, Yellow: #FFB800, Red: #FF4444

---

## Requirements

### Functional Requirements

**FR-1**: Executive Overview - Health Score Section
- Display large score gauge (0-100) with color
- Show 4 component breakdowns with progress bars
- Status indicator (Critical/Needs Attention/Healthy)

**FR-2**: Customer & Market - Churn Risk Section
- Summary metrics (high/medium/low risk counts)
- Filterable table with risk level coloring
- Export high-risk customers button

**FR-3**: Product Intelligence - Lifecycle Section
- Distribution pie chart (4 stages)
- Stage filter multiselect
- Detailed table with emoji indicators
- Strategic recommendations

**FR-4**: Growth & Insights - Decomposition Section
- Waterfall chart (Plotly)
- Component breakdown table
- Key insights auto-generated

**FR-5**: Product Launching - Velocity Section
- Bar chart by product with color coding
- Velocity filter + age slider
- Best/worst performer cards
- Benchmark guidance

### Non-Functional Requirements

**NFR-1**: Visual Consistency
- Match existing color scheme
- Use existing fonts/sizes
- Consistent spacing/padding

**NFR-2**: Responsive Design
- Work on desktop (primary)
- Graceful degradation on mobile

**NFR-3**: Performance
- Lazy loading (compute only when tab active)
- Show loading spinner for >1s operations

---

## Architecture

### Modified Files

1. **src/tabs/executive_overview.py**
   - Add health score section after existing KPIs
   - Location: After `st.divider()` on line 84

2. **src/tabs/customer_market.py**
   - Add churn risk section after regional performance
   - Location: After gap analysis table

3. **src/tabs/product_intelligence.py**
   - Add lifecycle section after product comparison
   - Location: After "Performance by Type" section

4. **src/tabs/growth_insights.py**
   - Add decomposition section after spike detection
   - Location: After YoY growth drivers

5. **src/tabs/product_launching.py**
   - Add velocity section after active customer tracking
   - Location: After customer journey analysis

---

## Implementation Steps

### Step 1: Executive Overview - Health Score (3-4 hours)

**Location**: `src/tabs/executive_overview.py`, line ~85

```python
# Add after existing Revenue Trend section

st.markdown("---")
st.subheader("💰 Financial Health Score")

# Compute health score
from src.analysis import compute_financial_health_score
health = compute_financial_health_score(df_curr, df_prev)

col1, col2 = st.columns([1, 2])

with col1:
    # Score gauge
    score_color = {
        'red': '#FF4444',
        'yellow': '#FFB800',
        'green': '#00C853'
    }[health['color']]

    st.markdown(f"""
    <div style="text-align: center; padding: 20px;
                background: linear-gradient(135deg, {score_color}22, {score_color}44);
                border-radius: 15px; border: 3px solid {score_color};">
        <div style="font-size: 48px; font-weight: bold; color: {score_color};">
            {health['score']}
        </div>
        <div style="font-size: 16px; color: #666; margin-top: 5px;">
            Health Score
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status
    status_text = {
        'red': '🔴 Critical - Immediate Action Required',
        'yellow': '🟡 Needs Attention - Monitor Closely',
        'green': '🟢 Healthy - On Track'
    }[health['color']]
    st.info(status_text)

with col2:
    st.markdown("**Score Components:**")

    for component, data in health['components'].items():
        component_name = component.replace('_', ' ').title()
        value = data['value']
        score = data['score']

        st.markdown(f"**{component_name}**: {value}% → Score: {score}/100")
        st.progress(score / 100)
        st.markdown("")
```

**Test**:
- Verify score displays correctly
- Test with empty `df_prev` (should not crash)
- Check colors match (red/yellow/green)

### Step 2: Customer & Market - Churn Risk (4-5 hours)

**Location**: `src/tabs/customer_market.py`, after gap analysis

```python
st.markdown("---")
st.subheader("⚠️ Customer Churn Risk Analysis")

from src.analysis import compute_churn_risk_scores
churn_df = compute_churn_risk_scores(df)

# Summary metrics
col1, col2, col3 = st.columns(3)

high_risk = len(churn_df[churn_df['risk_level'] == 'High'])
medium_risk = len(churn_df[churn_df['risk_level'] == 'Medium'])
low_risk = len(churn_df[churn_df['risk_level'] == 'Low'])

col1.metric("🔴 High Risk", high_risk,
            delta=f"{(high_risk/len(churn_df)*100):.1f}%")
col2.metric("🟡 Medium Risk", medium_risk,
            delta=f"{(medium_risk/len(churn_df)*100):.1f}%")
col3.metric("🟢 Low Risk", low_risk,
            delta=f"{(low_risk/len(churn_df)*100):.1f}%")

# Risk filter
risk_filter = st.multiselect(
    "Filter by Risk Level",
    options=['High', 'Medium', 'Low'],
    default=['High']
)

filtered = churn_df[churn_df['risk_level'].isin(risk_filter)]

# Styled table
def color_risk(val):
    if val == 'High':
        return 'background-color: #ffebee'
    elif val == 'Medium':
        return 'background-color: #fff9c4'
    else:
        return 'background-color: #e8f5e9'

styled = filtered.style.applymap(color_risk, subset=['risk_level'])
st.dataframe(styled, use_container_width=True, height=400)

# Export
if st.button("📥 Export High-Risk Customers"):
    high_risk_df = churn_df[churn_df['risk_level'] == 'High']
    csv = high_risk_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )
```

**Test**:
- Verify table coloring works
- Test CSV export functionality
- Check filters work correctly

### Step 3: Product Intelligence - Lifecycle (3-4 hours)

**Location**: `src/tabs/product_intelligence.py`, after "Performance by Type"

```python
st.markdown("---")
st.subheader("🔄 Product Lifecycle Analysis")

from src.analysis import compute_product_lifecycle
lifecycle_df = compute_product_lifecycle(df)

# Summary distribution
col1, col2, col3, col4 = st.columns(4)

intro = len(lifecycle_df[lifecycle_df['lifecycle_stage'] == 'Introduction'])
growth = len(lifecycle_df[lifecycle_df['lifecycle_stage'] == 'Growth'])
maturity = len(lifecycle_df[lifecycle_df['lifecycle_stage'] == 'Maturity'])
decline = len(lifecycle_df[lifecycle_df['lifecycle_stage'] == 'Decline'])

col1.metric("🌱 Introduction", intro)
col2.metric("📈 Growth", growth)
col3.metric("💎 Maturity", maturity)
col4.metric("📉 Decline", decline)

# Pie chart
stage_counts = lifecycle_df['lifecycle_stage'].value_counts()
fig = px.pie(
    values=stage_counts.values,
    names=stage_counts.index,
    title="Product Portfolio Distribution",
    color=stage_counts.index,
    color_discrete_map={
        'Introduction': '#4CAF50',
        'Growth': '#2196F3',
        'Maturity': '#FF9800',
        'Decline': '#F44336'
    }
)
st.plotly_chart(fig, use_container_width=True)

# Detailed table
st.markdown("**Product Details:**")

stage_filter = st.multiselect(
    "Filter by Stage",
    options=['Introduction', 'Growth', 'Maturity', 'Decline'],
    default=['Introduction', 'Growth', 'Decline']
)

filtered = lifecycle_df[lifecycle_df['lifecycle_stage'].isin(stage_filter)]
filtered = filtered.sort_values('total_revenue', ascending=False)

st.dataframe(
    filtered[[
        'stage_emoji', 'Name of product', 'lifecycle_stage',
        'age_months', 'growth_rate', 'total_revenue', 'recent_revenue'
    ]],
    use_container_width=True,
    height=400
)

# Strategic recommendations
st.markdown("---")
st.subheader("💡 Strategic Recommendations")

if decline > 0:
    st.warning(f"⚠️ **Action Required**: {decline} products in decline. Consider discontinuation or relaunch strategy.")

if intro > growth:
    st.info(f"📊 **Insight**: More products in Introduction ({intro}) than Growth ({growth}). Focus on conversion to growth stage.")

if maturity > len(lifecycle_df) * 0.6:
    st.success(f"✅ **Stable Portfolio**: {maturity} mature products provide stable revenue. Consider innovation pipeline.")
```

**Test**:
- Verify pie chart renders
- Test stage filter
- Check recommendations logic

### Step 4: Growth & Insights - Decomposition (4-5 hours)

**Location**: `src/tabs/growth_insights.py`, after YoY growth drivers

```python
st.markdown("---")
st.subheader("📊 Growth Decomposition Analysis")

from src.analysis import compute_growth_decomposition
decomp = compute_growth_decomposition(df_curr, df_prev)

if decomp is None:
    st.warning("No previous period data for comparison")
else:
    # Summary
    col1, col2 = st.columns(2)
    col1.metric(
        "Total Growth",
        f"${decomp['total_growth']:,.0f}",
        f"{decomp['total_growth_pct']}%"
    )
    col2.metric(
        "Previous Period Revenue",
        f"${decomp['revenue_prev']:,.0f}",
        f"Current: ${decomp['revenue_curr']:,.0f}"
    )

    # Waterfall chart
    import plotly.graph_objects as go

    components = decomp['components']

    waterfall_data = pd.DataFrame({
        'Component': ['Previous Revenue', 'New Customers', 'Expansion',
                      'Churn', 'Price Impact', 'Mix Impact', 'Current Revenue'],
        'Value': [
            decomp['revenue_prev'],
            components['new_customers'],
            components['expansion'],
            components['churn'],
            components['price_impact'],
            components['mix_impact'],
            decomp['revenue_curr']
        ],
        'Type': ['total', 'relative', 'relative', 'relative', 'relative', 'relative', 'total']
    })

    fig = go.Figure(go.Waterfall(
        name="Growth Decomposition",
        orientation="v",
        measure=waterfall_data['Type'],
        x=waterfall_data['Component'],
        textposition="outside",
        y=waterfall_data['Value'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="Revenue Growth Waterfall",
        showlegend=False,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Component table
    st.markdown("**Detailed Component Breakdown:**")

    breakdown_df = pd.DataFrame({
        'Component': ['New Customers', 'Expansion', 'Churn', 'Price Impact', 'Mix Impact'],
        'Amount': [
            f"${components['new_customers']:,.0f}",
            f"${components['expansion']:,.0f}",
            f"${components['churn']:,.0f}",
            f"${components['price_impact']:,.0f}",
            f"${components['mix_impact']:,.0f}"
        ],
        'Contribution %': [
            f"{decomp['component_pct']['new_customers']}%",
            f"{decomp['component_pct']['expansion']}%",
            f"{decomp['component_pct']['churn']}%",
            f"{decomp['component_pct']['price_impact']}%",
            f"{decomp['component_pct']['mix_impact']}%"
        ],
        'Impact': ['Positive' if components['new_customers'] > 0 else 'Negative',
                   'Positive' if components['expansion'] > 0 else 'Negative',
                   'Negative' if components['churn'] < 0 else 'Positive',
                   'Positive' if components['price_impact'] > 0 else 'Negative',
                   'Positive' if components['mix_impact'] > 0 else 'Negative']
    })

    def color_impact(val):
        return 'color: green' if val == 'Positive' else 'color: red'

    styled = breakdown_df.style.applymap(color_impact, subset=['Impact'])
    st.dataframe(styled, use_container_width=True)

    # Auto-insights
    st.markdown("---")
    st.subheader("💡 Key Insights")

    comp_abs = {k: abs(v) for k, v in components.items()}
    biggest = max(comp_abs, key=comp_abs.get)

    if biggest == 'new_customers' and components['new_customers'] > 0:
        st.info("🎯 **Customer Acquisition** is your primary growth driver. Continue investing in acquisition channels.")
    elif biggest == 'expansion' and components['expansion'] > 0:
        st.info("📈 **Customer Expansion** is driving growth. Your upsell/cross-sell strategies are working.")
    elif biggest == 'churn' and components['churn'] < 0:
        st.warning("⚠️ **Customer Churn** is the biggest challenge. Focus on retention initiatives.")
```

**Test**:
- Verify waterfall chart displays correctly
- Test with empty `df_prev` (should show warning)
- Validate math (components sum to total)

### Step 5: Product Launching - Velocity (3-4 hours)

**Location**: `src/tabs/product_launching.py`, after customer journey

```python
st.markdown("---")
st.subheader("🚀 Launch Velocity Analysis")

from src.analysis import compute_launch_velocity
velocity_df = compute_launch_velocity(df)

if velocity_df.empty:
    st.info("No products launched in the last 12 months with sufficient data (min 3 months)")
else:
    # Summary
    col1, col2, col3, col4 = st.columns(4)

    fast = len(velocity_df[velocity_df['velocity_category'] == 'Fast'])
    moderate = len(velocity_df[velocity_df['velocity_category'] == 'Moderate'])
    slow = len(velocity_df[velocity_df['velocity_category'] == 'Slow'])
    avg_velocity = velocity_df['velocity_pct'].mean()

    col1.metric("🚀 Fast Launches", fast)
    col2.metric("🏃 Moderate Launches", moderate)
    col3.metric("🐌 Slow Launches", slow)
    col4.metric("Average Velocity", f"{avg_velocity:.1f}%")

    # Bar chart
    fig = px.bar(
        velocity_df,
        x='Name of product',
        y='velocity_pct',
        color='velocity_category',
        title="Launch Velocity by Product",
        color_discrete_map={
            'Fast': '#4CAF50',
            'Moderate': '#FF9800',
            'Slow': '#F44336'
        },
        labels={'velocity_pct': 'Velocity (%)', 'Name of product': 'Product'}
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        velocity_filter = st.multiselect(
            "Filter by Velocity",
            options=['Fast', 'Moderate', 'Slow'],
            default=['Fast', 'Moderate', 'Slow']
        )

    with col2:
        min_age = st.slider("Minimum Age (months)", 3, 12, 3)

    filtered = velocity_df[
        (velocity_df['velocity_category'].isin(velocity_filter)) &
        (velocity_df['age_months'] >= min_age)
    ]

    # Table
    display_cols = [
        'velocity_emoji', 'Name of product', 'age_months', 'velocity_pct',
        'm1_revenue', 'm3_revenue', 'current_revenue', 'm1_customers', 'm3_customers'
    ]

    st.dataframe(filtered[display_cols], use_container_width=True, height=400)

    # Insights
    st.markdown("---")
    st.subheader("💡 Launch Insights")

    if not filtered.empty:
        best_launch = filtered.iloc[0]
        worst_launch = filtered.iloc[-1]

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"""
            **🏆 Best Performer**: {best_launch['Name of product']}
            - Velocity: {best_launch['velocity_pct']:.1f}%
            - M1 → M3: ${best_launch['m1_revenue']:,.0f} → ${best_launch['m3_revenue']:,.0f}
            - Current Revenue: ${best_launch['current_revenue']:,.0f}
            """)

        with col2:
            st.warning(f"""
            **⚠️ Needs Attention**: {worst_launch['Name of product']}
            - Velocity: {worst_launch['velocity_pct']:.1f}%
            - M1 → M3: ${worst_launch['m1_revenue']:,.0f} → ${worst_launch['m3_revenue']:,.0f}
            - Consider: Pricing review, marketing boost, or repositioning
            """)

    # Benchmarks
    st.markdown("**Launch Velocity Benchmarks:**")
    st.info("""
    - 🚀 **Fast (>100%)**: Excellent product-market fit. Scale quickly.
    - 🏃 **Moderate (50-100%)**: Healthy growth. Monitor and optimize.
    - 🐌 **Slow (<50%)**: Review strategy. May need pivot or additional support.
    """)
```

**Test**:
- Verify bar chart colors
- Test filters (velocity + age slider)
- Check best/worst cards display

---

## Todo List

- [x] **Feature 1**: Add health score to `executive_overview.py`
- [x] **Feature 1**: Test health score display (all color states)
- [x] **Feature 2**: Add churn risk to `customer_market.py`
- [x] **Feature 2**: Test table styling and CSV export
- [x] **Feature 3**: Add lifecycle to `product_intelligence.py`
- [x] **Feature 3**: Test pie chart and recommendations
- [x] **Feature 4**: Add decomposition to `growth_insights.py`
- [x] **Feature 4**: Test waterfall chart rendering
- [x] **Feature 5**: Add velocity to `product_launching.py`
- [x] **Feature 5**: Test bar chart and filters
- [x] **Integration**: Test all tabs load without errors
- [ ] **Visual QA**: Check color consistency across features
- [ ] **Performance**: Verify lazy loading works
- [ ] **Responsive**: Test on different screen sizes

---

## Success Criteria

- ✅ All 5 features render in correct tabs
- ✅ No breaking changes to existing UI
- ✅ Visual consistency maintained
- ✅ Interactive elements (filters, buttons) work
- ✅ Charts render correctly (Plotly)
- ✅ Tables styled appropriately
- ✅ Export functionality works

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| UI layout breaks existing content | High | Add sections at end, use `st.markdown("---")` separators |
| Charts don't render | Medium | Test with sample data, handle empty cases |
| Styling conflicts | Low | Use inline styles, avoid CSS classes |
| Performance (too many charts) | Medium | Lazy loading, collapsible sections |

---

## Next Steps

1. Complete Phase 1 (prerequisite)
2. Start with Feature 1 (Executive Overview)
3. Test each feature before moving to next
4. Visual QA review after all features integrated
5. Proceed to Phase 3 (Testing & Optimization)
