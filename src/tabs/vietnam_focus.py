import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_vietnam_focus(df, df_curr, df_prev, has_prev_year, current_year_val):
    st.subheader("🇻🇳 Focus TOP High & Less Performance in Local Vietnam")
    
    # Filter for Vietnam
    # Try to find 'Country' column. If not exists, check Region names?
    # User dataset likely has 'Country'.
    if 'Country' in df.columns:
        # Check values - User confirmed 'Viet Nam'
        # Using 'Viet' to be safe for both 'Vietnam' and 'Viet Nam'
        vn_mask_curr = df_curr['Country'].str.contains('Viet', case=False, na=False)
        vn_mask_prev = df_prev['Country'].str.contains('Viet', case=False, na=False) if not df_prev.empty else pd.Series([False]*len(df_prev))
        
        df_vn_curr = df_curr[vn_mask_curr]
        df_vn_prev = df_prev[vn_mask_prev] if not df_prev.empty else pd.DataFrame(columns=df_curr.columns)
        
        # If empty but Region has Vietnam?
        if df_vn_curr.empty:
            # Fallback to Region check
            vn_mask_curr = df_curr['Region'].str.contains('Viet', case=False, na=False)
            df_vn_curr = df_curr[vn_mask_curr]
            if not df_prev.empty:
                    vn_mask_prev = df_prev['Region'].str.contains('Viet', case=False, na=False)
                    df_vn_prev = df_prev[vn_mask_prev]
    else:
        st.warning("Column 'Country' not found. Creating focus based on available data.")
        df_vn_curr = df_curr
        df_vn_prev = df_prev

    if df_vn_curr.empty:
        st.info("No data found for 'Viet Nam' in the current selection.")
    else:
        # Helper to Render Category Focus
        def render_category_focus(cat_name, filter_keywords):
            st.markdown(f"#### 🏷️ {cat_name} Performance (Top 10)")
            
            # Filter Category using Regex OR
            # join keywords with |
            pattern = '|'.join([k.strip() for k in filter_keywords])
            
            mask_c = df_vn_curr['Type of product'].str.contains(pattern, case=False, na=False)
            mask_p = df_vn_prev['Type of product'].str.contains(pattern, case=False, na=False) if not df_vn_prev.empty else pd.Series([False]*len(df_vn_prev))
            
            d_c = df_vn_curr[mask_c]
            d_p = df_vn_prev[mask_p] if not df_vn_prev.empty else pd.DataFrame()
            
            if d_c.empty:
                st.write(f"No data found for {cat_name} (Keywords: {filter_keywords}).")
                return
            
            # Group by Name of Product
            top_c = d_c.groupby('Name of product')['Sold'].sum().reset_index().rename(columns={'Sold': 'Vol_Curr'})
            
            # 2024 (Prev)
            if not d_p.empty:
                top_p = d_p.groupby('Name of product')['Sold'].sum().reset_index().rename(columns={'Sold': 'Vol_Prev'})
            else:
                top_p = pd.DataFrame(columns=['Name of product', 'Vol_Prev'])
            
            # Sort Top 10 by Current Volume
            top_c = top_c.sort_values('Vol_Curr', ascending=False)
            
            top_10_names = top_c.head(10)['Name of product'].tolist()
            
            merged = pd.merge(top_c[top_c['Name of product'].isin(top_10_names)], top_p, on='Name of product', how='left').fillna(0)
            
            # Calc Delta
            merged['Delta %'] = ((merged['Vol_Curr'] - merged['Vol_Prev']) / merged['Vol_Prev']).replace([np.inf, -np.inf], 0).fillna(0)
            
            # Handle New
            new_mask = (merged['Vol_Prev'] == 0) & (merged['Vol_Curr'] > 0)
            merged.loc[new_mask, 'Delta %'] = 1.0
            
            # Sort descending volume 
            merged = merged.sort_values('Vol_Curr', ascending=False)
            
            # Table
            def color_delta_simple(val):
                color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else ''
                return f'background-color: {color}; color: black'

            # Define d_top10 here to be available for both Chart and Insights
            d_top10 = d_c[d_c['Name of product'].isin(top_10_names)].copy()
            d_top10['Short Name'] = d_top10['Name of product'].str.replace('ANDROS PROFESSIONAL', '', case=False).str.strip()
            d_top10['Short Name'] = d_top10['Short Name'].str.replace(r'\s+', ' ', regex=True)

            c_table, c_chart = st.columns([1, 1])
            
            with c_table:
                st.caption("Top 10 Comparison")
                st.dataframe(
                    merged.style.format({
                        "Vol_Curr": "{:,.0f}",
                        "Vol_Prev": "{:,.0f}",
                        "Delta %": "{:+.1%}"
                    }).map(color_delta_simple, subset=['Delta %']),
                    column_config={
                        "Name of product": "Product Name",
                        "Vol_Curr": f"Vol {current_year_val}",
                        "Vol_Prev": f"Vol {current_year_val - 1}" if has_prev_year else "Vol Prev",
                        "Delta %": "% Change"
                    },
                    use_container_width=True,
                    hide_index=True
                )
            
            with c_chart:
                st.caption("Regional Breakdown (Top 10)")
                
                # Macro Region Mapping
                def map_region(r):
                    r_u = r.upper()
                    if 'SOUTH' in r_u: return 'South'
                    if 'NORTH' in r_u: return 'North'
                    if 'CENTER' in r_u or 'CENTRAL' in r_u: return 'Center'
                    return 'Other'
                
                d_top10['Macro Region'] = d_top10['Region'].apply(map_region)
                
                # Group by Product and Macro Region for the chart
                d_chart_agg = d_top10.groupby(['Name of product', 'Macro Region'])['Sold'].sum().reset_index()
                
                # Define Colors: South=Green, North=Blue, Center=Orange
                region_colors = {
                    'South': '#2ca02c',  # Green
                    'North': '#1f77b4',  # Blue
                    'Center': '#ff7f0e', # Orange
                    'Other': '#7f7f7f'   # Grey
                }

                # Stacked 100% chart
                fig_stack = px.bar(
                    d_chart_agg, 
                    x='Name of product', 
                    y='Sold', 
                    color='Macro Region',
                    title=f"Regional Mix (100% Stacked)",
                    barmode='relative',
                    text_auto='.0f',
                    color_discrete_map=region_colors,
                    category_orders={"Macro Region": ["South", "North", "Center", "Other"]}
                )
                fig_stack.update_layout(
                    yaxis_title="% Volume", 
                    xaxis_title="Product", 
                    template="plotly_white", 
                    barnorm='percent',
                    font=dict(size=10), # Reduce global font size
                    xaxis=dict(tickfont=dict(size=9)), # Smaller x-axis labels
                    uniformtext_minsize=8, uniformtext_mode='hide'
                )
                st.plotly_chart(fig_stack, use_container_width=True)
        
            # --- Automated Regional Insights ---
            if not d_top10.empty and 'Kind of fruit' in d_top10.columns:
                interest_df = d_top10.groupby(['Kind of fruit', 'Macro Region'])['Sold'].sum().reset_index()
                
                # Pivot to find max region per fruit
                pivot_interest = interest_df.pivot(index='Kind of fruit', columns='Macro Region', values='Sold').fillna(0)
                
                # Calculate mapped preferences
                preferences = []
                for fruit in pivot_interest.index:
                    vols = pivot_interest.loc[fruit]
                    total = vols.sum()
                    if total == 0: continue
                    
                    winner = vols.idxmax()
                    pct = vols[winner] / total
                    
                    preferences.append((fruit, winner, pct))
                
                # Group by Region to construct sentence
                region_fruits = {}
                for fruit, region, pct in preferences:
                    if region not in region_fruits: region_fruits[region] = []
                    region_fruits[region].append(f"{fruit} ({pct:.0%})")
                
                if region_fruits:
                    insight_lines = []
                    for reg, fruits in region_fruits.items():
                        if reg == 'Other': continue
                        fruit_list = ", ".join(fruits)
                        insight_lines.append(f"- **{reg}**: Drivers include {fruit_list}.")
                    
                    st.info(f"💡 **Regional Interest Insights ({cat_name}):**\n\n" + "\n".join(insight_lines))
                else:
                    st.info(f"No clear regional dominance found for top {cat_name} items.")
            else:
                    pass
        
        # Categories Definition
        bev_types = ['Chunky', 'Fruit Mix', 'Jelly Top Up', 'Syrup']
        bakery_types = ['Frozen Puree', 'Fruit Filling', 'Frozen Fruit', 'Top&Fill', 'Top & Fill', 'Chocofill'] 
        
        # Render Sections
        render_category_focus("Beverage", bev_types)
        st.divider()
        render_category_focus("Bakery", bakery_types)
