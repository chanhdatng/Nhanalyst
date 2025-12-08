import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils import DEFAULT_DATE_COL

def render_customer_market(df, df_curr):
    c1, c2 = st.columns(2)
        
    with c1:
        st.subheader("🌍 Regional Performance")
        region_stats = df_curr.groupby('Country')['Sold'].sum().reset_index().sort_values('Sold', ascending=False)
        fig_map = px.bar(region_stats, x='Sold', y='Country', orientation='h', title="Revenue by Country", text_auto=',.0f')
        fig_map.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_map, use_container_width=True)
        
    with c2:
        st.subheader("👥 Client Segments (RFM Approximation)")
        # Very simple segmentation based on Revenue
        client_rev = df_curr.groupby('Name of client')['Sold'].sum().reset_index()
        def segment(row):
            if row['Sold'] > 50000: return 'Diamond' # arbitrary thresholds
            elif row['Sold'] > 10000: return 'Gold'
            elif row['Sold'] > 1000: return 'Silver'
            else: return 'Bronze'
        
        if not client_rev.empty:
            client_rev['Segment'] = client_rev.apply(segment, axis=1)
            seg_counts = client_rev['Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig_pie = px.pie(seg_counts, names='Segment', values='Count', title="Client Value Distribution", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            with st.expander("ℹ️ Segmentation Criteria"):
                st.markdown("""
                - **Diamond**: > 50,000 KG
                - **Gold**: > 10,000 KG
                - **Silver**: > 1,000 KG
                - **Bronze**: ≤ 1,000 KG
                """)
        else:
            st.write("No client data available.")
    
    st.subheader("🏆 Top Clients")
    
    # Channel Filter
    avail_channels = sorted(df['Channel by Sales Person'].dropna().unique()) if 'Channel by Sales Person' in df.columns else []
    selected_channels = st.multiselect("Filter by Channel (Sales Person):", options=avail_channels, default=avail_channels)
    
    if selected_channels:
            df_clients_src = df_curr[df_curr['Channel by Sales Person'].isin(selected_channels)]
    else:
            df_clients_src = df_curr
    
    top_clients = df_clients_src.groupby('Name of client').agg(
        Total_Revenue=('Sold', 'sum'),
        Orders=('Sold', 'count'),
        Last_Order_Month=('Month', 'max')
    ).sort_values('Total_Revenue', ascending=False).head(20).reset_index()

    # --- 6-Month Gap Analysis Columns ---
    if not df.empty:
            max_d = df[DEFAULT_DATE_COL].max()
            cutoff_gap = max_d - pd.DateOffset(months=6)
            
            # Filter for 6m & Channel
            df_6m_gap = df[(df[DEFAULT_DATE_COL] >= cutoff_gap) & (df[DEFAULT_DATE_COL] <= max_d)]
            if selected_channels:
                df_6m_gap = df_6m_gap[df_6m_gap['Channel by Sales Person'].isin(selected_channels)]
            
            # Universe: All fruits sold in this period/channel per Type
            type_universe = df_6m_gap.groupby('Type of product')['Kind of fruit'].unique().apply(set).to_dict()
            
            type_col_data = []
            gap_col_data = []
            
            for client in top_clients['Name of client']:
                client_data = df_6m_gap[df_6m_gap['Name of client'] == client]
                
                if client_data.empty:
                    type_col_data.append("No 6m Data")
                    gap_col_data.append("-")
                    continue
                
                # Types
                bought_types = sorted(client_data['Type of product'].dropna().unique())
                type_str = ", ".join(bought_types)
                type_col_data.append(type_str)
                
                # Variety Gaps
                details = []
                for t in bought_types:
                    if t not in type_universe: continue
                    
                    bought_fruits = set(client_data[client_data['Type of product'] == t]['Kind of fruit'].dropna().unique())
                    all_vals = type_universe[t]
                    missing = sorted(list(all_vals - bought_fruits))
                    
                    # Format: "Puree: ✅3 ❌5 (Miss: Mango...)"
                    miss_txt = ", ".join(missing[:3])
                    if len(missing) > 3: miss_txt += "..."
                    if not missing: miss_txt = "-"
                    
                    line = f"**{t}**: ✅{len(bought_fruits)} / ❌{len(missing)} (Miss: {miss_txt})"
                    details.append(line)
                
                gap_col_data.append("\n".join(details))
            
            top_clients['Types of Product (6m)'] = type_col_data
            top_clients['Fruit Variety Analysis (6m)'] = gap_col_data
    
    st.dataframe(
        top_clients.style.format({"Total_Revenue": "{:,.0f}", "Orders": "{:,.0f}"}),
        column_config={
            "Total_Revenue": st.column_config.NumberColumn("Total Volume (KG)"),
            "Orders": st.column_config.NumberColumn(),
            "Types of Product (6m)": st.column_config.TextColumn(width="medium"),
            "Fruit Variety Analysis (6m)": st.column_config.TextColumn(width="large"),
        },
        use_container_width=True, 
        hide_index=True
    )
