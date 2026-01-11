import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils import calculate_growth, DEFAULT_DATE_COL

def render_executive_overview(df_curr, df_prev, selected_years, selected_months, current_year_val, single_year_mode, has_prev_year):
    comp_label = "vs Prev Year" if single_year_mode and has_prev_year else ("vs Prev Month" if single_year_mode else "Multi-Year View")

    st.subheader(f"Key Performance Indicators ({comp_label if has_prev_year else 'MoM (Last Month)'})")
    
    # Calculate Metrics
    curr_rev = df_curr['Sold'].sum()
    curr_vol = df_curr['Quantity (KG)'].sum()
    curr_clients = df_curr['Name of client'].nunique()
    curr_orders = len(df_curr)
    curr_aov = curr_rev / curr_orders if curr_orders else 0
    
    # Growth Calculation
    growth_rev, growth_vol, growth_aov, growth_clients = None, None, None, None
    
    if has_prev_year:
        # YoY
        prev_rev = df_prev['Sold'].sum()
        prev_vol = df_prev['Quantity (KG)'].sum()
        prev_orders = len(df_prev)
        prev_aov = prev_rev / prev_orders if prev_orders else 0
        prev_clients = df_prev['Name of client'].nunique()
        
        growth_rev = calculate_growth(curr_rev, prev_rev)
        growth_vol = calculate_growth(curr_vol, prev_vol)
        growth_aov = calculate_growth(curr_aov, prev_aov)
        growth_clients = calculate_growth(curr_clients, prev_clients)
        
    elif not df_curr.empty:
        # Single Year / No Prev Year -> MoM Calculation (Last Month vs Month Before)
        # Aggregate by month to find latest month
        monthly_agg = df_curr.groupby('Month').agg({
            'Sold': 'sum', 
            'Quantity (KG)': 'sum',
            'Name of client': 'nunique'
        }).sort_index()
        
        # Use 'date__ym' to sort correctly if Month col is messy
        monthly_agg_t = df_curr.groupby(DEFAULT_DATE_COL).agg({
            'Sold': 'sum', 
            'Quantity (KG)': 'sum',
            'Name of client': 'nunique',
            'Month': 'count' # orders proxy
        }).sort_index()
        
        if len(monthly_agg_t) >= 2:
            last_m = monthly_agg_t.iloc[-1]
            prev_m = monthly_agg_t.iloc[-2]
            
            # Show growth of LATEST MONTH in the selection
            # Note: The Displayed "Total Revenue" is typically TOTAL for the selection.
            # Showing growth of "Total 2024" vs "Total 2023(missing)" is N/A.
            # But showing growth of "Dec" vs "Nov" implies the KPI card is for "Dec".
            # If user selected "All", Total is Year Total.
            
            # Compromise:
            # If "All" selected -> Show Total. Growth is N/A (or maybe show trend sparkline?).
            # If "Single Month" selected -> Show MoM.
            
            if selected_months and len(selected_months) == 1:
                # Valid MoM case
                prev_rev = 0 
                # Try to find prev month in full data?
                # Too complex for this snippet as we don't have full df passed just for this edge case in strict structure,
                # but we could assume df_curr is enough if we had more context. 
                # For now, pass.
                pass
    
    # Display Metrics (removed Total Revenue since Sold = Volume not monetary)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Volume (KG)", f"{curr_vol:,.0f}", f"{growth_vol:.1%}" if growth_vol else None)
    c2.metric("Avg Order Size (KG)", f"{curr_aov:,.0f}", f"{growth_aov:.1%}" if growth_aov else None)
    c3.metric("Total Clients", f"{curr_clients:,.0f}", f"{growth_clients:.1%}" if growth_clients else None)
    
    if not has_prev_year:
        st.caption("ℹ️ Previous Year data not available. Comparisons are disabled.")
    
    st.divider()
    
    # Trend Analysis
    st.subheader("Revenue Trend")
    
    if len(selected_years) > 1:
        # Multi-year selection: Distinct lines for each year
        trend_curr = df_curr.sort_values(DEFAULT_DATE_COL).groupby(['Year', 'Month'])['Sold'].sum().reset_index()
        trend_curr['Type'] = trend_curr['Year'].astype(str)
        trend_combined = trend_curr
        title_chart = "Monthly Revenue Trend (Multi-Year)"
    else:
        # Single Less selection: Aggregate by Month + Optional YoY
        def agg_monthly(d):
            d = d.sort_values(DEFAULT_DATE_COL)
            return d.groupby('Month', sort=False)['Sold'].sum().reset_index()
        
        trend_curr = agg_monthly(df_curr).assign(Type='Selected Period')
        
        if has_prev_year and not df_prev.empty:
            trend_prev = agg_monthly(df_prev).assign(Type=f'{current_year_val - 1}')
            trend_curr['Type'] = f'{current_year_val}' # Rename for clarity
            trend_combined = pd.concat([trend_curr, trend_prev])
            title_chart = "Monthly Revenue Comparison (YoY)"
        else:
            trend_combined = trend_curr
            title_chart = "Monthly Revenue Trend"
    
    # Chart Controls
    chart_type = st.radio("Chart Type", ["Bar", "Line"], horizontal=True, key="monthly_chart_type")

    if chart_type == "Bar":
        fig_trend = px.bar(
            trend_combined, 
            x='Month', 
            y='Sold', 
            color='Type', 
            barmode='group',
            color_discrete_map={f'{current_year_val}': '#1E90FF', f'{current_year_val - 1}': '#D3D3D3'} if single_year_mode else None,
            title=title_chart
        )
    else:
        fig_trend = px.line(
            trend_combined, 
            x='Month', 
            y='Sold', 
            color='Type', 
            markers=True,
            text='Sold',
            color_discrete_map={f'{current_year_val}': '#1E90FF', f'{current_year_val - 1}': '#D3D3D3'} if single_year_mode else None,
            title=title_chart
        )
        fig_trend.update_traces(textposition="top center", texttemplate="%{text:,.0f}")

    fig_trend.update_layout(xaxis_title="Month", yaxis_title="Revenue", template="plotly_white", margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_trend, use_container_width=True)

    # ==========================================================================
    # PHASE 2: Financial Health Score Section
    # ==========================================================================
    st.markdown("---")
    st.subheader("💰 Financial Health Score")
    
    from src.analysis import compute_financial_health_score
    health = compute_financial_health_score(df_curr, df_prev)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Score gauge with color
        score_color = {
            'red': '#FF4444',
            'yellow': '#FFB800',
            'green': '#00C853'
        }.get(health['color'], '#666')
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;
                    background: #ffffff;
                    border-radius: 15px; border: 3px solid {score_color};">
            <div style="font-size: 48px; font-weight: bold; color: {score_color};">
                {health['score']}
            </div>
            <div style="font-size: 16px; color: #333333; margin-top: 5px;">
                Health Score
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicator
        status_text = {
            'red': '🔴 Critical - Immediate Action Required',
            'yellow': '🟡 Needs Attention - Monitor Closely',
            'green': '🟢 Healthy - On Track'
        }.get(health['color'], '⚪ Unknown')
        
        st.info(status_text)
    
    with col2:
        st.markdown("**Score Components:**")
        
        component_labels = {
            'volume_growth': '📈 Volume Growth',
            'avg_order_size': '📦 Avg Order Size (KG)',
            'retention': '🔄 Customer Retention',
            'order_size_growth': '📊 Order Size Growth'
        }
        
        for component, data in health['components'].items():
            label = component_labels.get(component, component.replace('_', ' ').title())
            value = data['value']
            score = data['score']
            weight = int(data['weight'] * 100)
            
            # Format value based on component type
            if component in ['volume_growth', 'order_size_growth', 'retention']:
                value_str = f"{value:+.1f}%" if component != 'retention' else f"{value:.1f}%"
            else:
                value_str = f"{value:,.0f} KG"
            
            st.markdown(f"**{label}** ({weight}% weight): {value_str} → Score: {score:.0f}/100")
            st.progress(min(score / 100, 1.0))

