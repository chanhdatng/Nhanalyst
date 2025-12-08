import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils import DEFAULT_DATE_COL

def render_product_launching(df, df_curr, df_prev, current_year_val):
    st.subheader("🚀 Product Launching Analysis")
    
    # User Feedback: Multi-select checkboxes + Apply button
    # Using st.form to batch updates
    with st.form("launch_form"):
        c_filt1, c_filt2 = st.columns(2)
        with c_filt1:
            # Use global df for stable options
            avail_types = sorted(df['Type of product'].dropna().unique()) if 'Type of product' in df.columns else []
            sel_types_launch = st.multiselect("Filter Type of Product:", options=avail_types, key="launch_filter_type")
            
        with c_filt2:
            # Show ALL Kinds to avoid dependency issues inside form
            avail_kinds = sorted(df['Kind of fruit'].dropna().unique()) if 'Kind of fruit' in df.columns else []
            sel_kinds_launch = st.multiselect("Filter Kind of Fruit:", options=avail_kinds, key="launch_filter_kind")
        
        st.form_submit_button("Run Analysis")

    if df_curr.empty:
        st.warning("No data for current selection.")
    else:
        # Logic: Identify Products Launched in this Year (First sale date in this Year)
        
        # Start with all in current
        df_potential = df_curr.copy()
        
        # Apply Filters
        if sel_types_launch:
            df_potential = df_potential[df_potential['Type of product'].isin(sel_types_launch)]
        if sel_kinds_launch:
            df_potential = df_potential[df_potential['Kind of fruit'].isin(sel_kinds_launch)]
        
        # Identify "Launched" (New)
        curr_prods = set(df_potential['Name of product'].unique())
        num_kinds = df_potential['Kind of fruit'].nunique()
        
        # If we have previous year data, determine which are new
        launched_prods = set()
        if not df_prev.empty:
            prev_prods = set(df_prev['Name of product'].unique())
            launched_prods = curr_prods - prev_prods
            launch_msg = f"Showing {len(curr_prods)} products ({num_kinds} Fruit Varieties). Found {len(launched_prods)} NEW products (not sold in {current_year_val - 1})."
        else:
            launched_prods = curr_prods # All considered new/launch if no prev data
            launch_msg = f"Showing {len(curr_prods)} products ({num_kinds} Fruit Varieties). (No previous year for comparison)."
        
        st.success(launch_msg)
        
        # USE ALL Products for analysis, not just launched
        df_launch = df_potential.copy()
            
        # --- 1. Launching Table ---

        # --- Dialog Function (Defined once) ---
        @st.dialog("Active Customers Details")
        def show_active_customers(type_cv, kind_cv=None):
            st.write(f"**Product Type:** {type_cv}")
            if kind_cv:
                st.write(f"**Kind:** {kind_cv}")
            
            # Re-calculate active list for this specific selection
            
            # 1. Broad Filter 6m
            if not df_launch.empty:
                max_d = df_launch[DEFAULT_DATE_COL].max()
            else:
                max_d = df[DEFAULT_DATE_COL].max()
            cutoff = max_d - pd.DateOffset(months=6)
            
            d_6m = df[(df[DEFAULT_DATE_COL] >= cutoff) & (df[DEFAULT_DATE_COL] <= max_d)]
            
            # 2. Specific Filter
            if kind_cv:
                mask = (d_6m['Type of product'] == type_cv) & (d_6m['Kind of fruit'] == kind_cv)
            else:
                mask = (d_6m['Type of product'] == type_cv)
            
            d_target = d_6m[mask]
            
            if d_target.empty:
                st.warning("No data found for this period.")
                return

            # 3. Group by Client
            stats = d_target.groupby('Name of client').agg(
                Total_Orders=('Name of client', 'count'),
                Total_Vol=('Sold', 'sum'),
                Last_Order=(DEFAULT_DATE_COL, 'max')
            ).reset_index()
            
            # 4. Filter Active (>= 2 orders)
            active_stats = stats[stats['Total_Orders'] >= 2].sort_values('Total_Vol', ascending=False)
            
            if active_stats.empty:
                    st.info("No active customers (>= 2 orders) found.")
            else:
                    st.write(f"Found **{len(active_stats)}** active customers (>= 2 orders in last 6 months).")
                    st.dataframe(
                        active_stats.style.format({
                            "Total_Vol": "{:,.1f}",
                            "Last_Order": "{:%Y-%m-%d}"
                        }),
                        column_config={
                            "Name of client": "Customer",
                            "Total_Orders": "Orders",
                            "Total_Vol": "Vol (KG)",
                            "Last_Order": "Last Purchase"
                        },
                        use_container_width=True,
                        hide_index=True
                    )

        # Custom Gradient
        def highlight_total(s):
            return ['background-color: #f0fdf4; color: #166534; font-weight: bold'] * len(s)

        def custom_greens(s):
            if s.empty: return [''] * len(s)
            s_num = pd.to_numeric(s, errors='coerce').fillna(0)
            min_val = s_num.min()
            max_val = s_num.max()
            rng = max_val - min_val if max_val != min_val else 1
            colors = []
            for val in s_num:
                norm = (val - min_val) / rng
                alpha = 0.1 + (0.9 * norm)
                colors.append(f'background-color: rgba(40, 167, 69, {alpha:.2f}); color: black')
            return colors

        # Optional Checkbox for Grouping
        group_by_type = st.checkbox("Group by Type of Product", value=True, key="launch_group_type")
        group_by_kind = st.checkbox("Group by Kind of Fruit", value=True, key="launch_group_fruit")
        
        # --- Global Totals for "Total Row" Resolution ---
        if not df_launch.empty:
            max_d_global = df_launch[DEFAULT_DATE_COL].max()
        else:
            max_d_global = df[DEFAULT_DATE_COL].max()
        cutoff_global = max_d_global - pd.DateOffset(months=6)
        
        # Global 6m History (Unfiltered by Region, but time-bound)
        df_6m_raw = df[(df[DEFAULT_DATE_COL] >= cutoff_global) & (df[DEFAULT_DATE_COL] <= max_d_global)]
        
        # 1. Total Row Calculation
        df_6m_context = df_6m_raw.copy()
        if sel_types_launch:
            df_6m_context = df_6m_context[df_6m_context['Type of product'].isin(sel_types_launch)]
        if sel_kinds_launch:
            df_6m_context = df_6m_context[df_6m_context['Kind of fruit'].isin(sel_kinds_launch)]
            
        # Identify Clients who are Active (>=2 orders) in this context
        client_order_counts = df_6m_context.groupby('Name of client').size()
        global_active_clients = set(client_order_counts[client_order_counts >= 2].index)
        
        # Current View Clients
        current_clients = set(df_launch['Name of client'].unique())
        
        # Intersection
        global_unique_cust = len(current_clients)
        global_active_cust = len(current_clients & global_active_clients)
        
        
        # --- Dynamic Region Pivot ---
        if group_by_type:
            # --- C. Type Level (Broadest) ---
            
            region_pivot = df_launch.pivot_table(index=['Type of product'], 
                                                    columns='Region', 
                                                    values='Sold', 
                                                    aggfunc='sum', 
                                                    fill_value=0).reset_index()
            
            region_cols = [c for c in region_pivot.columns if c != 'Type of product']
            
            cust_counts = df_launch.groupby(['Type of product'])['Name of client'].nunique().reset_index().rename(columns={'Name of client': 'Number of customers'})
            
            launch_clients_type = df_launch[['Type of product', 'Name of client']].drop_duplicates()
            
            target_types = df_launch['Type of product'].unique()
            df_6m_type = df_6m_raw[df_6m_raw['Type of product'].isin(target_types)]
            
            type_client_counts = df_6m_type.groupby(['Type of product', 'Name of client']).size().reset_index(name='PriorOrders')
            active_pairs = type_client_counts[type_client_counts['PriorOrders'] >= 2]
            
            active_in_view = pd.merge(launch_clients_type, active_pairs, on=['Type of product', 'Name of client'], how='inner')
            
            active_counts = active_in_view.groupby('Type of product')['Name of client'].nunique().reset_index().rename(columns={'Name of client': 'Active Customers'})
            
            counts_merged = pd.merge(cust_counts, active_counts, on='Type of product', how='left')
            counts_merged['Active Customers'] = counts_merged['Active Customers'].fillna(0).astype(int)
            
            final_table = pd.merge(region_pivot, counts_merged, on=['Type of product'], how='left')
            
            final_table['Total'] = final_table[region_cols].sum(axis=1)
            
            cols_order = ['Type of product'] + sorted(region_cols) + ['Total', 'Number of customers', 'Active Customers']
            final_table = final_table[cols_order].sort_values('Total', ascending=False).reset_index(drop=True)
            
            st.caption("Select a row to view Active Customers details.")
            
            numeric_cols = region_cols + ['Total', 'Number of customers', 'Active Customers']
            
            event = st.dataframe(
                final_table.style.format("{:,.0f}", subset=numeric_cols).apply(highlight_total, subset=['Total']),
                column_config={
                    "Name of client": "Customer",
                },
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            if event.selection.rows:
                idx = event.selection.rows[0]
                selected_row = final_table.iloc[idx]
                show_active_customers(selected_row['Type of product'], None)

            # Total Row
            sum_cols = region_cols + ['Total']
            total_values = final_table[sum_cols].sum()
            
            full_sum_cols = sum_cols + ['Number of customers', 'Active Customers']
            total_data = {c: total_values[c] for c in sum_cols}
            total_data['Type of product'] = 'TOTAL'
            total_data['Number of customers'] = global_unique_cust
            total_data['Active Customers'] = global_active_cust
            
            total_row = pd.DataFrame([total_data])
            total_row = total_row[cols_order]
            
            st.dataframe(
                total_row.style.format("{:,.0f}", subset=full_sum_cols).apply(highlight_total, subset=['Total']),
                use_container_width=True,
                hide_index=True
            )


        elif group_by_kind:
                # --- B. Kind Level (Grouped) ---
                
            region_pivot = df_launch.pivot_table(index=['Type of product', 'Kind of fruit'], 
                                                    columns='Region', 
                                                    values='Sold', 
                                                    aggfunc='sum', 
                                                    fill_value=0).reset_index()
            
            region_cols = [c for c in region_pivot.columns if c not in ['Type of product', 'Kind of fruit']]
            
            cust_counts = df_launch.groupby(['Type of product', 'Kind of fruit'])['Name of client'].nunique().reset_index().rename(columns={'Name of client': 'Number of customers'})
            
            launch_clients_kind = df_launch[['Type of product', 'Kind of fruit', 'Name of client']].drop_duplicates()
            
            target_keys = df_launch[['Type of product', 'Kind of fruit']].drop_duplicates()
            df_6m_kind = pd.merge(df_6m_raw, target_keys, on=['Type of product', 'Kind of fruit'], how='inner')
            
            kind_client_counts = df_6m_kind.groupby(['Type of product', 'Kind of fruit', 'Name of client']).size().reset_index(name='PriorOrders')
            active_pairs_kind = kind_client_counts[kind_client_counts['PriorOrders'] >= 2]
            
            active_in_view = pd.merge(launch_clients_kind, active_pairs_kind, on=['Type of product', 'Kind of fruit', 'Name of client'], how='inner')
            active_counts = active_in_view.groupby(['Type of product', 'Kind of fruit'])['Name of client'].nunique().reset_index().rename(columns={'Name of client': 'Active Customers'})
            
            counts_merged = pd.merge(cust_counts, active_counts, on=['Type of product', 'Kind of fruit'], how='left')
            counts_merged['Active Customers'] = counts_merged['Active Customers'].fillna(0).astype(int)
            
            final_table = pd.merge(region_pivot, counts_merged, on=['Type of product', 'Kind of fruit'], how='left')
            final_table['Total'] = final_table[region_cols].sum(axis=1)
            
            cols_order = ['Type of product', 'Kind of fruit'] + sorted(region_cols) + ['Total', 'Number of customers', 'Active Customers']
            final_table = final_table[cols_order].sort_values('Total', ascending=False).reset_index(drop=True)
            
            st.caption("Select a row to view Active Customers details.")
            
            numeric_cols = region_cols + ['Total', 'Number of customers', 'Active Customers']
            
            event = st.dataframe(
                final_table.style.format("{:,.0f}", subset=numeric_cols).apply(highlight_total, subset=['Total']),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            if event.selection.rows:
                idx = event.selection.rows[0]
                selected_row = final_table.iloc[idx]
                show_active_customers(selected_row['Type of product'], selected_row['Kind of fruit'])
            
            # Total Row
            sum_cols = region_cols + ['Total']
            total_values = final_table[sum_cols].sum()
            
            full_sum_cols = sum_cols + ['Number of customers', 'Active Customers']
            total_data = {c: total_values[c] for c in sum_cols}
            total_data['Type of product'] = 'TOTAL'
            total_data['Kind of fruit'] = ''
            total_data['Number of customers'] = global_unique_cust
            total_data['Active Customers'] = global_active_cust
            
            total_row = pd.DataFrame([total_data])
            total_row = total_row[cols_order]
            
            st.dataframe(
                total_row.style.format("{:,.0f}", subset=full_sum_cols).apply(highlight_total, subset=['Total']),
                use_container_width=True,
                hide_index=True
            )

        else:
            # --- A. Product Level (Default) ---
            
            region_pivot = df_launch.pivot_table(index=['Type of product', 'Kind of fruit', 'Name of product'], 
                                                    columns='Region', 
                                                    values='Sold', 
                                                    aggfunc='sum', 
                                                    fill_value=0).reset_index()
            
            region_cols = [c for c in region_pivot.columns if c not in ['Type of product', 'Kind of fruit', 'Name of product']]
            
            region_pivot['Total'] = region_pivot[region_cols].sum(axis=1)
            
            final_table = region_pivot.sort_values('Total', ascending=False).reset_index(drop=True)
            
            # Calculate Total Row
            sum_cols = region_cols + ['Total']
            total_values = final_table[sum_cols].sum()
            
            # Reconstruct Total Row to match
            cols_display = ['Type of product', 'Kind of fruit', 'Name of product'] + sorted(region_cols) + ['Total']
            final_table = final_table[cols_display]
            
            total_row_disp = pd.DataFrame(columns=cols_display)
            total_row_disp.loc[0, 'Type of product'] = 'TOTAL'
            for c in sum_cols:
                total_row_disp.loc[0, c] = total_values[c]
            total_row_disp = total_row_disp.fillna('')

            numeric_cols = region_cols + ['Total']

            st.dataframe(
                final_table.style.format("{:,.0f}", subset=numeric_cols).apply(highlight_total, subset=['Total']),
                column_config={
                    "Type of product": "Type",
                    "Kind of fruit": "Kind",
                    "Name of product": "Product",
                },
                use_container_width=True,
                hide_index=True
            )     
            
            st.dataframe(
                total_row_disp.style.format("{:,.0f}", subset=sum_cols).apply(highlight_total, subset=['Total']),
                use_container_width=True,
                hide_index=True
            )          
        # --- 2. Customer List Expander ---
        with st.expander("View Customer Details for Launched Products"):
            if df_launch.empty:
                st.info("No data available.")
            else:
                # Create Short Label
                df_launch['Prod_Label'] = df_launch['Type of product'] + " - " + df_launch['Kind of fruit']
                
                # Identify Top 10 Products (Labels) by Volume
                top_prod_labels = df_launch.groupby('Prod_Label')['Sold'].sum().sort_values(ascending=False).head(10).index.tolist()
                
                # Filter for Top 10
                df_launch_sub_cust = df_launch[df_launch['Prod_Label'].isin(top_prod_labels)].copy()
                
                # --- Dynamic Logic: Split by Year if few products ---
                unique_years_cust = df_launch_sub_cust['Year'].unique()
                if len(top_prod_labels) < 5 and len(unique_years_cust) > 1:
                    df_launch_sub_cust['Prod_Label'] = df_launch_sub_cust['Prod_Label'] + " (" + df_launch_sub_cust['Year'].astype(str) + ")"
                
                # Pivot
                cust_matrix = df_launch_sub_cust.pivot_table(
                    index=['Region', 'Name of client'], 
                    columns='Prod_Label', 
                    values='Sold', 
                    aggfunc='sum', 
                    fill_value=0
                )
                
                cust_matrix['Total'] = cust_matrix.sum(axis=1)
                cust_matrix = cust_matrix.sort_values('Total', ascending=False)
                
                # Add Total Row (Grand Total)
                grand_total = cust_matrix.sum()
                total_row_df = pd.DataFrame([grand_total], columns=cust_matrix.columns)
                total_row_df.index = pd.MultiIndex.from_tuples([('TOTAL', '')], names=['Region', 'Name of client'])
                cust_matrix = pd.concat([total_row_df, cust_matrix])
                
                st.dataframe(
                    cust_matrix.style.format("{:,.0f}").apply(custom_greens, subset=['Total']),
                    use_container_width=True
                )
        
        st.divider()
        
        # --- 3. Charts ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Monthly Sales Trend")
            
            # Local Filters for Trend Chart Only
            with st.expander("Filter Trend Data (Year / Customer)", expanded=False):
                ft_col1, ft_col2 = st.columns(2)
                with ft_col1:
                        # Re-create broad context for Options
                        df_launch_all = df.copy()
                        if sel_types_launch:
                            df_launch_all = df_launch_all[df_launch_all['Type of product'].isin(sel_types_launch)]
                        if sel_kinds_launch:
                            df_launch_all = df_launch_all[df_launch_all['Kind of fruit'].isin(sel_kinds_launch)]

                        avail_years_trend = sorted(df_launch_all['Year'].unique())
                        if len(avail_years_trend) > 1:
                            sel_years_trend = st.multiselect("Select Year:", list(map(int, avail_years_trend)), default=list(map(int, avail_years_trend)), key="trend_filter_year")
                        else:
                            sel_years_trend = list(map(int, avail_years_trend))
                with ft_col2:
                        avail_clients_trend = sorted(df_launch_all['Name of client'].dropna().unique())
                        sel_clients_trend = st.multiselect("Select Customer:", avail_clients_trend, key="trend_filter_client")

            df_trend_src = df_launch.copy()
            
            # Apply Local Filters
            if len(avail_years_trend) > 1 and sel_years_trend:
                df_trend_src = df_trend_src[df_trend_src['Year'].isin(sel_years_trend)]
            if sel_clients_trend:
                df_trend_src = df_trend_src[df_trend_src['Name of client'].isin(sel_clients_trend)]
            
            if df_trend_src.empty:
                st.warning("No data for Trend Chart after filtering.")
            else:
                if sel_clients_trend:
                    trend_options = sorted(df_launch_all[df_launch_all['Name of client'].isin(sel_clients_trend)]['Name of product'].unique())
                else:
                    trend_options = sorted(df_launch_all['Name of product'].unique())
                
                select_all_prods = st.checkbox("Select All Products", key="trend_select_all")
                if select_all_prods:
                    sel_trend_prods = trend_options
                    st.caption("All products selected.")
                else:
                        sel_trend_prods = st.multiselect("Select Products for Trend (Default: All):", options=trend_options, key="trend_filter_product")
                
                color_col = None
                
                # --- Multi-Year Handling ---
                multi_year_trend = (len(df_trend_src['Year'].unique()) > 1)
                df_trend_src['Label'] = df_trend_src['Name of product'] # Default Label
                
                if sel_trend_prods:
                    df_trend_src = df_trend_src[df_trend_src['Name of product'].isin(sel_trend_prods)]
                    
                    if multi_year_trend:
                        df_trend_src['Label'] = df_trend_src['Name of product'] + " (" + df_trend_src['Year'].astype(str) + ")"
                        color_col = 'Label'
                        group_cols = ['Year', 'Month', 'Name of product', 'Label']
                    else:
                        color_col = 'Name of product'
                        group_cols = ['Month', 'Name of product']
                        
                    title_text = "Monthly Traction (Selected)"
                else:
                    # All Products Aggregated
                    if multi_year_trend:
                        df_trend_src['Label'] = df_trend_src['Year'].astype(str)
                        color_col = 'Label'
                        group_cols = ['Year', 'Month', 'Label']
                    else:    
                        group_cols = ['Month']
                    title_text = "Monthly Traction (All Launched)"

                # Grouping
                trend = df_trend_src.groupby(group_cols)['Sold'].sum().reset_index()
                
                # Check column mapping
                if color_col and color_col not in trend.columns:
                     # e.g. if we grouped by Month only
                     color_col = None

                fig_trend = px.line(trend, x='Month', y='Sold', color=color_col, markers=True, title=title_text, text='Sold')
                fig_trend.update_traces(texttemplate='%{text:,.0f}', textposition='top center')
                fig_trend.update_layout(template="plotly_white")
                st.plotly_chart(fig_trend, use_container_width=True)
                
            with c2:
                st.subheader("Regional Distribution")
                # Stacked 100% Column Chart with % Labels
                
                # Determine Main Column based on final_table content
                if 'Name of product' in final_table.columns:
                    main_col = 'Name of product'
                    chart_title_suffix = "Top 15 New Products"
                elif 'Kind of fruit' in final_table.columns:
                    main_col = 'Kind of fruit'
                    chart_title_suffix = "Top 15 Kinds of Fruit"
                else:
                    main_col = 'Type of product'
                    chart_title_suffix = "Product Types"
                
                # Top 15 by Volume for readability
                top_launch = final_table.head(15)[main_col].tolist()
                
                # Filter source data
                df_launch_sub = df_launch[df_launch[main_col].isin(top_launch)].copy()
                
                # Clean names for chart
                if main_col == 'Name of product':
                    df_launch_sub['Short Name'] = df_launch_sub['Name of product'].str.replace('ANDROS PROFESSIONAL', '', case=False).str.strip()
                    df_launch_sub['Short Name'] = df_launch_sub['Short Name'].str.replace(r'\s+', ' ', regex=True)
                else:
                    df_launch_sub['Short Name'] = df_launch_sub[main_col]
                
                # Chart Aggregation (Sum over selected years/regions)
                chart_agg = df_launch_sub.groupby(['Short Name', 'Region'])['Sold'].sum().reset_index()
                
                # Calc Percentage
                chart_agg['Total_Prod'] = chart_agg.groupby('Short Name')['Sold'].transform('sum')
                chart_agg['Pct'] = chart_agg['Sold'] / chart_agg['Total_Prod']
                chart_agg['Label'] = chart_agg['Pct'].apply(lambda x: f"{x:.0%}" if x > 0.05 else "") # Hide small labels
                
                fig_stack = px.bar(
                    chart_agg,
                    x='Short Name',
                    y='Sold',
                    color='Region',
                    title=f"Regional Mix ({chart_title_suffix})",
                    text='Label'
                )
                fig_stack.update_layout(template="plotly_white", barnorm='percent', yaxis_title="% Volume", xaxis_title=main_col)
                st.plotly_chart(fig_stack, use_container_width=True)
