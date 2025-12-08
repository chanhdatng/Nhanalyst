import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.utils import filter_by_date

def render_growth_insights(df_curr, df_prev, selected_years, selected_months, current_year_val, single_year_mode, has_prev_year, df, selected_regions):
    st.subheader("🚀 Growth Intelligence & Spike Analysis")
    
    # --- Section 1: Type of Product Analysis ---
    st.write("### 1. Product Type Performance")
    st.caption("Analyze which Product Types (e.g., FROZEN PUREE, FROZEN FRUIT) are driving volume.")
    
    # Group by Type of Product
    type_stats = df_curr.groupby('Type of product').agg(
        Volume=('Sold', 'sum'),
        Kind_Count=('Kind of fruit', 'nunique')
    ).reset_index().sort_values('Volume', ascending=False)
    
    c_type1, c_type2 = st.columns([1, 2])
    
    with c_type1:
        st.dataframe(
            type_stats.style.format({"Volume": "{:,.0f}"}), 
            column_config={
                "Volume": st.column_config.NumberColumn(),
                "Kind_Count": st.column_config.NumberColumn("Fruit Varieties", help="Number of distinct fruits"),
            },
            hide_index=True,
            use_container_width=True
        )
        
    with c_type2:
        # Trend by Type
        # Filter top 5 types to avoid clutter
        top_types = type_stats.head(5)['Type of product'].tolist()
        # df_curr is full data
        df_type_trend = df_curr[df_curr['Type of product'].isin(top_types)].copy()
        
        # Multi-Year Logic
        if len(selected_years) > 1:
            df_type_trend['Label'] = df_type_trend['Type of product'] + " (" + df_type_trend['Year'].astype(str) + ")"
            type_trend = df_type_trend.groupby(['Year', 'Month', 'Type of product', 'Label'])['Sold'].sum().reset_index()
            color_col = 'Label'
        else:
            type_trend = df_type_trend.groupby(['Month', 'Type of product'])['Sold'].sum().reset_index()
            color_col = 'Type of product'
        
        # Ensure sorting for line chart
        type_trend = type_trend.sort_values(['Month'])
        
        fig_type = px.line(
            type_trend, x='Month', y='Sold', color=color_col, 
            title='Monthly Trend by Top Product Types', markers=True,
            text='Sold'
        )
        fig_type.update_traces(textposition="top center", texttemplate="%{text:,.0f}")
        st.plotly_chart(fig_type, use_container_width=True)

    st.divider()

    # --- Section 2: Spike Detection ---
    st.write("### 2. Spike Detection (Growth > 30%)")
    st.caption("Identify SKUs that surged in volume compared to the previous period and see WHO bought them.")
    
    spike_df = pd.DataFrame()
    
    # single_year_mode passed as arg
    if not single_year_mode:
        st.warning("⚠️ Spike detection requires a Single Year selection to perform clear Period-over-Period comparison.")
    else:
        c_ctrl1, c_ctrl2 = st.columns([1, 2])
        with c_ctrl1:
            spike_mode = st.radio("Compare vs:", ["Previous Year (YoY)", "Previous Month (MoM)"], horizontal=True)
        with c_ctrl2:
            # Add Min Volume Threshold
            min_vol = st.number_input("Ignore Spikes with Current Volume < (KG):", min_value=0, value=50, step=10)
        
        baseline_df = pd.DataFrame()
        # current_year_val passed
        
        error_msg = None

        if spike_mode == "Previous Year (YoY)":
            years_in_data = df['Year'].unique()
            prev_yr = current_year_val - 1
            if prev_yr in years_in_data:
                baseline_df = filter_by_date(df, [prev_yr], selected_months)
                baseline_df = baseline_df[baseline_df['Region'].isin(selected_regions)]
            else:
                error_msg = "No Previous Year data available for YoY comparison."
        else: # MoM
            all_months = sorted(df['Month'].dropna().unique())
            # Logic: If 1 month selected (e.g. 5), baseline is month 4.
            if selected_months and len(selected_months) == 1:
                    try:
                        # Assume selected_months has values matching 'Month' col which should be INTs now due to cleaning
                        curr_m_val = selected_months[0]
                        # Find prev value in all_months list
                        if curr_m_val in all_months:
                            idx = list(all_months).index(curr_m_val)
                            if idx > 0:
                                prev_m_val = all_months[idx - 1]
                                baseline_df = filter_by_date(df, [current_year_val], [prev_m_val])
                                baseline_df = baseline_df[baseline_df['Region'].isin(selected_regions)]
                            else:
                                error_msg = "Selected month is the first available. Cannot compare to previous."
                    except Exception as e:
                        error_msg = f"Error determining previous month: {e}"
            else:
                error_msg = "For MoM Spike detection, please select exactly ONE month in the sidebar."

        if error_msg:
            st.warning(f"⚠️ {error_msg}")
        
        elif not baseline_df.empty:
            # Calculate Growth per SKU
            curr_sku = df_curr.groupby(['SKU', 'Name of product'])['Sold'].sum().reset_index().rename(columns={'Sold': 'Curr_Vol'})
            base_sku = baseline_df.groupby(['SKU', 'Name of product'])['Sold'].sum().reset_index().rename(columns={'Sold': 'Base_Vol'})
            
            merged = pd.merge(curr_sku, base_sku, on=['SKU', 'Name of product'], how='left').fillna(0)
            
            merged['Growth_Pct'] = (merged['Curr_Vol'] - merged['Base_Vol']) / merged['Base_Vol']
            # Clean up infinite/NaN
            merged['Growth_Pct'] = merged['Growth_Pct'].replace([np.inf, -np.inf], 0).fillna(0) 
                # Logic adjustment: If Base is 0 and Curr > 0, growth is Infinite ideally, 
                # but technically undefined numerator relative to 0. 
                # Let's handle: if Base=0 & Curr>0 => New Listing or Re-introduction.
                # Let's treat as High Growth (1.0 or 100% just for sorting)?
                # Actually, let's mark as 100% (1.0) if base is 0. 
            mask_new = (merged['Base_Vol'] == 0) & (merged['Curr_Vol'] > 0)
            merged.loc[mask_new, 'Growth_Pct'] = 1.0 # Treat as 100% growth for sorting
            
            # Threshold > 30% AND Min Volume
            spikes = merged[
                (merged['Growth_Pct'] > 0.3) & 
                (merged['Curr_Vol'] >= min_vol)
            ].sort_values('Growth_Pct', ascending=False)
            
            if spikes.empty:
                st.success(f"No SKUs with >30% growth (and >{min_vol}kg) found ({spike_mode}).")
            else:
                st.success(f"Found {len(spikes)} SKUs with >30% growth!")
                
                # 1. Select SKU
                sku_labels = spikes.apply(
                    lambda x: f"{x['Name of product']} (SKU: {x['SKU']}) | +{x['Growth_Pct']:.1%} ({x['Base_Vol']:,.0f}kg ➡️ {x['Curr_Vol']:,.0f}kg)", 
                    axis=1
                ).tolist()
                selected_spike_label = st.selectbox("Select SKU to Analyze:", options=sku_labels)
                
                if selected_spike_label:
                    # Extract SKU
                    import re
                    match = re.search(r'SKU:\s*(\d+)', selected_spike_label)
                    if match:
                        sel_sku_id = int(match.group(1))
                        
                        # 2. Drill Down
                        st.markdown(f"**Recall: Who bought this SKU?**")
                        
                        # Filter Main DF for this SKU in Current Period
                        # Cast sel_sku_id to string because SKU col is cleaned as string
                        sku_sales = df_curr[df_curr['SKU'] == str(sel_sku_id)]
                        
                        client_contrib = sku_sales.groupby('Name of client')['Sold'].sum().reset_index().sort_values('Sold', ascending=False)
                        total_sku_vol = client_contrib['Sold'].sum()
                        client_contrib['Share'] = client_contrib['Sold'] / total_sku_vol if total_sku_vol else 0
                        
                        st.dataframe(
                            client_contrib.head(10).style.format({"Sold": "{:,.1f}"}),
                            column_config={
                                "Sold": st.column_config.NumberColumn("Volume (KG)"),
                                "Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1)
                            },
                            hide_index=True,
                            use_container_width=True
                        )

    st.divider()

    # --- Section 3: Advanced YoY Dynamics ---
    st.write("### 3. Advanced YoY Dynamics")
    st.caption("Detailed Comparison: Current Year vs Previous Year (requires single previous year data).")
    
    if not has_prev_year:
            st.info("ℹ️ Please select a Single Year that has a Previous Year available in the data to unlock YoY Dynamics.")
    else:
        # Helper to render the analysis for a given dataframe subset
        def render_yoy_analysis_helper(d_curr, d_prev, label_suffix):
            if d_curr.empty and d_prev.empty:
                st.write("No data available for this category.")
                return

            # 3.1 Fruit Performance Matrix
            st.markdown(f"#### 🍎 Kind of Fruit ({label_suffix}): {current_year_val} vs {current_year_val - 1}")
            
            # Curr Agg
            fruit_curr = d_curr.groupby('Kind of fruit').agg(
                Vol_Curr=('Sold', 'sum'),
                Clients_Curr=('Name of client', 'nunique')
            ).reset_index()
            
            # Prev Agg
            fruit_prev = d_prev.groupby('Kind of fruit')['Sold'].sum().reset_index().rename(columns={'Sold': 'Vol_Prev'})
            
            # Merge
            fruit_yoy = pd.merge(fruit_curr, fruit_prev, on='Kind of fruit', how='outer').fillna(0)
            
            fruit_yoy['Delta'] = fruit_yoy['Vol_Curr'] - fruit_yoy['Vol_Prev']
            fruit_yoy['Growth %'] = (fruit_yoy['Delta'] / fruit_yoy['Vol_Prev']).replace([np.inf, -np.inf], 0).fillna(0)
            
            # Handle Infinite growth where Vol_Prev = 0 and Vol_Curr > 0
            new_fruit_mask = (fruit_yoy['Vol_Prev'] == 0) & (fruit_yoy['Vol_Curr'] > 0)
            fruit_yoy.loc[new_fruit_mask, 'Growth %'] = 1.0 
            
            # Sort by Delta Descending (Winners first)
            fruit_yoy = fruit_yoy.sort_values('Delta', ascending=False)
            
            def color_delta(val):
                color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else ''
                return f'background-color: {color}; color: black'

            st.dataframe(
                fruit_yoy.style.format({
                    "Vol_Curr": "{:,.0f}",
                    "Vol_Prev": "{:,.0f}",
                    "Delta": "{:+,.0f}",
                    "Growth %": "{:+.1%}",
                    "Clients_Curr": "{:,.0f}"
                }).map(color_delta, subset=['Delta']),
                column_config={
                    "Kind of fruit": "Fruit Variety",
                    "Vol_Curr": f"Vol {current_year_val}",
                    "Vol_Prev": f"Vol {current_year_val - 1}",
                    "Clients_Curr": "Active Clients (Curr)"
                },
                use_container_width=True,
                hide_index=True
            )
            
            # 3.2 Monthly Spike Drivers (>20% YoY)
            st.markdown(f"#### ⚡ Monthly Trend Drivers ({label_suffix})")
            st.caption(f"Analyzing months in **{current_year_val}** with **>20% YoY Growth** caused by New vs Existing Clients.")
            
            with st.expander("ℹ️ Method Definitions"):
                st.markdown("""
                - **New Clients**: Clients with sales in the current month but NO sales in the same month last year.
                - **Lost Clients**: Clients with sales in the same month last year but NO sales in the current month.
                - **Existing Clients**: Clients with sales in BOTH periods. Only the *change* in volume (Current - Previous) is shown.
                """)
            
            # --- Specific Fruit Filter for this Section ---
            avail_fruits_yoy = sorted(d_curr['Kind of fruit'].dropna().unique())
            sel_fruits_yoy = st.multiselect(f"Filter Fruit Variety ({label_suffix}):", options=avail_fruits_yoy, default=avail_fruits_yoy)
            
            if not sel_fruits_yoy:
                    st.warning("Please select at least one Fruit Variety.")
            else:
                # Apply Filter
                d_curr_f = d_curr[d_curr['Kind of fruit'].isin(sel_fruits_yoy)]
                d_prev_f = d_prev[d_prev['Kind of fruit'].isin(sel_fruits_yoy)] if not d_prev.empty else d_prev
                
                # 1. Identify Months
                m_curr = d_curr_f.groupby('Month')['Sold'].sum().reset_index().rename(columns={'Sold': 'Vol_Curr'})
                m_prev = d_prev_f.groupby('Month')['Sold'].sum().reset_index().rename(columns={'Sold': 'Vol_Prev'})
                
                if not m_prev.empty:
                    m_merged = pd.merge(m_curr, m_prev, on='Month', how='inner') 
                    m_merged['Growth'] = (m_merged['Vol_Curr'] - m_merged['Vol_Prev']) / m_merged['Vol_Prev']
                else:
                    m_merged = m_curr.copy()
                    m_merged['Vol_Prev'] = 0
                    m_merged['Growth'] = 1.0 # 100% growth if no prev
            
                
                spike_months = m_merged[m_merged['Growth'] > 0.20]['Month'].tolist()
                
                if not spike_months:
                    st.info("No months found with >20% YoY growth.")
                else:
                    # Tabs for each month
                    m_tabs = st.tabs([f"Month {m}" for m in spike_months])
                    
                    for tab, m in zip(m_tabs, spike_months):
                        with tab:
                            # Driver Analysis Logic
                            d_c = d_curr_f[d_curr_f['Month'] == m]
                            d_p = d_prev_f[d_prev_f['Month'] == m]
                            
                            clients_c = set(d_c['Name of client'].unique())
                            clients_p = set(d_p['Name of client'].unique())
                            
                            new_clients = clients_c - clients_p
                            lost_clients = clients_p - clients_c
                            existing_clients = clients_c.intersection(clients_p)
                            
                            # Volumes
                            vol_new = d_c[d_c['Name of client'].isin(new_clients)]['Sold'].sum()
                            vol_lost = d_p[d_p['Name of client'].isin(lost_clients)]['Sold'].sum()
                            
                            # Existing Expansion/Contraction
                            vol_ex_prev = d_p[d_p['Name of client'].isin(existing_clients)]['Sold'].sum()
                            vol_ex_curr = d_c[d_c['Name of client'].isin(existing_clients)]['Sold'].sum()
                            vol_existing_delta = vol_ex_curr - vol_ex_prev
                            
                            total_delta = d_c['Sold'].sum() - d_p['Sold'].sum()
                            
                            bridge_data = pd.DataFrame([
                                {"Factor": "New Clients", "Impact (KG)": vol_new, "Type": "Positive"},
                                {"Factor": "Existing Clients (Expansion)", "Impact (KG)": vol_existing_delta, "Type": "Positive" if vol_existing_delta >=0 else "Negative"},
                                {"Factor": "Lost Clients", "Impact (KG)": -vol_lost, "Type": "Negative"}
                            ])
                            
                            fig_bridge = px.bar(
                                bridge_data, x="Factor", y="Impact (KG)", color="Type", 
                                text="Impact (KG)",
                                color_discrete_map={"Positive": "green", "Negative": "red"},
                                title=f"YoY Growth Drivers: Month {m}"
                            )

                            fig_bridge.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                            st.plotly_chart(fig_bridge, use_container_width=True)
                            
                            st.markdown(f"**Total Net Increase:** {total_delta:+,.0f} KG")
                            
                            # --- Display Detailed Client Lists ---
                            st.markdown("#### 📋 Client Detail Lists")
                            c_d1, c_d2, c_d3 = st.columns(3)
                            
                            with c_d1:
                                st.markdown("**🆕 New Clients**")
                                if new_clients:
                                    # New Clients: Show Current Volume
                                    df_new = d_c[d_c['Name of client'].isin(new_clients)].groupby('Name of client')['Sold'].sum().reset_index().sort_values('Sold', ascending=False)
                                    st.dataframe(
                                        df_new.style.format({"Sold": "{:,.0f}"}),
                                        column_config={
                                            "Name of client": "Client",
                                            "Sold": st.column_config.NumberColumn("Vol (Curr)")
                                        },
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                else:
                                    st.caption("No new clients.")

                            with c_d2:
                                st.markdown("**🔻 Lost Clients**")
                                if lost_clients:
                                    # Lost Clients: Show Previous Volume (what was lost)
                                    df_lost = d_p[d_p['Name of client'].isin(lost_clients)].groupby('Name of client')['Sold'].sum().reset_index().sort_values('Sold', ascending=False)
                                    st.dataframe(
                                        df_lost.style.format({"Sold": "{:,.0f}"}),
                                        column_config={
                                            "Name of client": "Client",
                                            "Sold": st.column_config.NumberColumn("Vol (Prev)")
                                        },
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                else:
                                    st.caption("No lost clients.")
                            
                            with c_d3:
                                st.markdown("**🔄 Existing Clients**")
                                if existing_clients:
                                    # Existing: Show Delta
                                    e_curr = d_c[d_c['Name of client'].isin(existing_clients)].groupby('Name of client')['Sold'].sum()
                                    e_prev = d_p[d_p['Name of client'].isin(existing_clients)].groupby('Name of client')['Sold'].sum()
                                    
                                    df_ex = pd.DataFrame({'Vol_Curr': e_curr, 'Vol_Prev': e_prev}).fillna(0)
                                    df_ex['Delta'] = df_ex['Vol_Curr'] - df_ex['Vol_Prev']
                                    df_ex = df_ex.sort_values('Delta', ascending=False).reset_index()
                                    
                                    def color_delta_ex(val):
                                        color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else ''
                                        return f'background-color: {color}; color: black'

                                    st.dataframe(
                                        df_ex.style.format({
                                            "Vol_Curr": "{:,.0f}",
                                            "Vol_Prev": "{:,.0f}",
                                            "Delta": "{:+,.0f}"
                                        }).map(color_delta_ex, subset=['Delta']),
                                        column_config={
                                            "Name of client": "Client",
                                            "Vol_Curr": "Curr",
                                            "Vol_Prev": "Prev",
                                            "Delta": "Change"
                                        },
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                else:
                                    st.caption("No existing clients.")

        # Create Selection for Product Types (User requested Dropdown instead of Tabs)
        avail_types = sorted(df_curr['Type of product'].dropna().unique())
        
        if not avail_types:
            st.warning("No Product Types found in current data.")
        else:
            p_type = st.selectbox("Select Product Type for YoY Analysis:", avail_types)
            
            # Filter Data for this specific Type
            mask_curr = df_curr['Type of product'] == p_type
            d_type_curr = df_curr[mask_curr]
            
            if not df_prev.empty:
                mask_prev = df_prev['Type of product'] == p_type
                d_type_prev = df_prev[mask_prev]
            else:
                d_type_prev = pd.DataFrame(columns=df_curr.columns)
                
            render_yoy_analysis_helper(d_type_curr, d_type_prev, p_type)
