import streamlit as st
import pandas as pd
import plotly.express as px

def render_product_intelligence(df_curr, selected_years, df=None):
    st.subheader("📦 Product Performance Analysis")
    
    # 1. Advanced Product Table
    prod_stats = df_curr.groupby(['Name of product', 'SKU']).agg(
        Volume=('Sold', 'sum'),  # Sold column is Volume in KG
        Orders=('Sold', 'count'),
        Clients=('Name of client', 'nunique')  # Number of unique clients
    ).reset_index()
    
    # Add contribution % (0-100 scale for ProgressColumn)
    total_vol = prod_stats['Volume'].sum()
    prod_stats['Contribution'] = (prod_stats['Volume'] / total_vol * 100) if total_vol else 0
    prod_stats = prod_stats.sort_values('Volume', ascending=False)
    
    st.dataframe(
        prod_stats.style.format({
            "Volume": "{:,.0f}"
        }),
        column_config={
            "Name of product": st.column_config.TextColumn("Product"),
            "Volume": st.column_config.TextColumn("Volume (KG)"),
            "Orders": st.column_config.NumberColumn("Orders"),
            "Clients": st.column_config.NumberColumn("Clients"),
            "Contribution": st.column_config.ProgressColumn(
                "Contribution %",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # 2. Comparator Tool
    st.subheader("📈 Product Comparison Tool")
    st.caption("Select products to compare their performance over time.")
    
    top_products = prod_stats.head(10)['Name of product'].tolist()
    selected_prods = st.multiselect("Choose Products to Compare", options=sorted(df_curr['Name of product'].unique()), default=top_products[:2] if len(top_products)>=2 else top_products)
    
    if selected_prods:
        # Filter
        df_comp = df_curr[df_curr['Name of product'].isin(selected_prods)].copy()
        
        # Multi-year Logic
        if len(selected_years) > 1:
            df_comp['Label'] = df_comp['Name of product'] + " (" + df_comp['Year'].astype(str) + ")"
            comp_trend = df_comp.groupby(['Year', 'Month', 'Name of product', 'Label'])['Sold'].sum().reset_index()
            color_col = 'Label'
        else:
            comp_trend = df_comp.groupby(['Month', 'Name of product'])['Sold'].sum().reset_index()
            color_col = 'Name of product'
        
        fig_comp = px.line(
            comp_trend, 
            x='Month', 
            y='Sold', 
            color=color_col, 
            markers=True,
            text='Sold',
            title="Revenue Comparison Trend"
        )
        fig_comp.update_traces(textposition="top center", texttemplate="%{text:,.0f}")
        fig_comp.update_layout(template="plotly_white")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Select products to compare.")

    # --- Section 3: Performance by Product Type ---
    st.subheader("📊 Performance by Type")

    # ==========================================================================
    # PHASE 2: Product Lifecycle Analysis Section
    # ==========================================================================
    
    # Use the full df if provided, otherwise use df_curr
    lifecycle_data = df if df is not None and not df.empty else df_curr
    
    st.markdown("---")
    st.subheader("🔄 Product Lifecycle Analysis")
    
    from src.analysis import compute_product_lifecycle
    lifecycle_df = compute_product_lifecycle(lifecycle_data)
    
    if lifecycle_df.empty:
        st.info("No product data available for lifecycle analysis.")
    else:
        # Summary distribution metrics
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
            },
            hole=0.4
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Stage filter
        st.markdown("**Product Details:**")
        
        stage_filter = st.multiselect(
            "Filter by Stage",
            options=['Introduction', 'Growth', 'Maturity', 'Decline'],
            default=['Introduction', 'Growth', 'Decline'],
            key="lifecycle_stage_filter"
        )
        
        if stage_filter:
            filtered = lifecycle_df[lifecycle_df['lifecycle_stage'].isin(stage_filter)].copy()
        else:
            filtered = lifecycle_df.copy()
        
        filtered = filtered.sort_values('total_volume', ascending=False)
        
        # Display columns
        display_cols = ['stage_emoji', 'Name of product', 'lifecycle_stage',
                       'age_months', 'growth_rate', 'total_volume', 'recent_volume']
        
        st.dataframe(
            filtered[display_cols].head(30).style.format({
                "growth_rate": "{:.1f}%",
                "total_volume": "{:,.0f}",
                "recent_volume": "{:,.0f}"
            }),
            column_config={
                "stage_emoji": st.column_config.TextColumn("", width="small"),
                "Name of product": st.column_config.TextColumn("Product"),
                "lifecycle_stage": st.column_config.TextColumn("Stage"),
                "age_months": st.column_config.NumberColumn("Age (Months)"),
                "growth_rate": st.column_config.TextColumn("Growth %"),
                "total_volume": st.column_config.TextColumn("Total Volume (KG)"),
                "recent_volume": st.column_config.TextColumn("Recent Vol (3M)"),
            },
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Strategic recommendations
        st.markdown("---")
        st.subheader("💡 Strategic Recommendations")
        
        total_products = len(lifecycle_df)
        
        if decline > 0:
            st.warning(f"⚠️ **Action Required**: {decline} product(s) in decline phase. Consider discontinuation or relaunch strategy.")
        
        if intro > growth:
            st.info(f"📊 **Insight**: More products in Introduction ({intro}) than Growth ({growth}). Focus on conversion to growth stage.")
        
        if maturity > total_products * 0.6:
            st.success(f"✅ **Stable Portfolio**: {maturity} mature products ({maturity/total_products*100:.0f}% of portfolio) provide stable volume. Consider building innovation pipeline.")
        
        if growth > total_products * 0.3:
            st.success(f"🚀 **Strong Growth**: {growth} products ({growth/total_products*100:.0f}%) in growth phase. Capitalize on momentum.")

