import streamlit as st
import pandas as pd
import plotly.express as px

def render_product_intelligence(df_curr, selected_years):
    st.subheader("📦 Product Performance Analysis")
    
    # 1. Advanced Product Table
    prod_stats = df_curr.groupby(['Name of product', 'SKU']).agg(
        Revenue=('Sold', 'sum'),
        Volume=('Quantity (KG)', 'sum'),
        Orders=('Sold', 'count')
    ).reset_index()
    
    # Add contribution %
    total_rev = prod_stats['Revenue'].sum()
    prod_stats['Contribution'] = prod_stats['Revenue'] / total_rev if total_rev else 0
    prod_stats = prod_stats.sort_values('Revenue', ascending=False)
    
    st.dataframe(
        prod_stats.style.format({"Revenue": "{:,.0f}", "Volume": "{:,.0f}"}),
        column_config={
            "Revenue": st.column_config.NumberColumn(),
            "Volume": st.column_config.NumberColumn(format="%f KG"), 
            "Contribution": st.column_config.ProgressColumn(
                format="%.1f%%",
                min_value=0,
                max_value=1
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
