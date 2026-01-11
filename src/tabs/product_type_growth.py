import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils import DEFAULT_DATE_COL


def render_product_type_growth(df, df_curr, selected_years):
    """
    Tab: Product Type Growth Analysis
    Shows Type of Product with:
    - Sold quantity (KG) for each selected year
    - % Growth YoY
    - Number of clients ordering each type
    - Active clients (>=2 orders in last 3 months AND >=1 order in previous month)
    """
    st.subheader("📈 Product Type Growth Analysis")

    if df_curr.empty:
        st.warning("No data for current selection.")
        return

    # Sort years for consistent display
    years_sorted = sorted(selected_years)

    # =========================================================================
    # SECTION 1: Summary Table
    # =========================================================================
    st.markdown("### 📊 Summary by Type of Product")

    # Build pivot: Type of product vs Year -> Sold (KG)
    pivot_data = df_curr.groupby(['Type of product', 'Year'])['Sold'].sum().reset_index()
    pivot_table = pivot_data.pivot(
        index='Type of product',
        columns='Year',
        values='Sold'
    ).fillna(0)

    # Rename columns to show year clearly
    pivot_table.columns = [f'Sold (KG) {int(y)}' for y in pivot_table.columns]
    pivot_table = pivot_table.reset_index()

    # Calculate % Growth (between consecutive years if multiple selected)
    year_cols = [c for c in pivot_table.columns if c.startswith('Sold (KG)')]

    if len(year_cols) >= 2:
        prev_col = year_cols[-2]
        curr_col = year_cols[-1]
        prev_year = prev_col.replace('Sold (KG) ', '')
        curr_year = curr_col.replace('Sold (KG) ', '')

        pivot_table['% Growth'] = pivot_table.apply(
            lambda row: ((row[curr_col] - row[prev_col]) / row[prev_col] * 100)
            if row[prev_col] > 0 else (100.0 if row[curr_col] > 0 else 0.0),
            axis=1
        )
        growth_label = f"% Growth ({prev_year} → {curr_year})"
        pivot_table = pivot_table.rename(columns={'% Growth': growth_label})
        growth_col = growth_label
    else:
        growth_col = None

    # Count unique clients per Type of product
    client_counts = df_curr.groupby('Type of product')['Name of client'].nunique().reset_index()
    client_counts.columns = ['Type of product', 'No. of Clients']

    # --- Calculate Active Clients ---
    max_date = df[DEFAULT_DATE_COL].max()
    cutoff_3m = max_date - pd.DateOffset(months=3)
    cutoff_prev_month = max_date - pd.DateOffset(months=1)

    df_3m = df[df[DEFAULT_DATE_COL] >= cutoff_3m]
    df_prev_month = df[(df[DEFAULT_DATE_COL] >= cutoff_prev_month) & (df[DEFAULT_DATE_COL] <= max_date)]

    active_clients_per_type = []
    for prod_type in pivot_table['Type of product'].unique():
        type_3m = df_3m[df_3m['Type of product'] == prod_type]
        client_order_counts = type_3m.groupby('Name of client').size()
        clients_2plus_orders = set(client_order_counts[client_order_counts >= 2].index)

        type_prev = df_prev_month[df_prev_month['Type of product'] == prod_type]
        clients_prev_month = set(type_prev['Name of client'].unique())

        active_clients = clients_2plus_orders & clients_prev_month
        active_clients_per_type.append({
            'Type of product': prod_type,
            'Active Clients': len(active_clients)
        })

    active_df = pd.DataFrame(active_clients_per_type)

    # Merge all data
    final_table = pivot_table.merge(client_counts, on='Type of product', how='left')
    final_table = final_table.merge(active_df, on='Type of product', how='left')
    final_table['No. of Clients'] = final_table['No. of Clients'].fillna(0).astype(int)
    final_table['Active Clients'] = final_table['Active Clients'].fillna(0).astype(int)

    if year_cols:
        final_table = final_table.sort_values(year_cols[-1], ascending=False)

    # --- Calculate TOTAL row ---
    total_row = {'Type of product': 'TOTAL'}
    for col in year_cols:
        total_row[col] = final_table[col].sum()
    if growth_col:
        prev_total = total_row[year_cols[-2]] if len(year_cols) >= 2 else 0
        curr_total = total_row[year_cols[-1]] if year_cols else 0
        total_row[growth_col] = ((curr_total - prev_total) / prev_total * 100) if prev_total > 0 else 0.0
    total_row['No. of Clients'] = df_curr['Name of client'].nunique()

    all_type_3m = df_3m[df_3m['Type of product'].isin(pivot_table['Type of product'])]
    all_client_order_counts = all_type_3m.groupby('Name of client').size()
    all_clients_2plus = set(all_client_order_counts[all_client_order_counts >= 2].index)
    all_type_prev = df_prev_month[df_prev_month['Type of product'].isin(pivot_table['Type of product'])]
    all_clients_prev = set(all_type_prev['Name of client'].unique())
    total_active = len(all_clients_2plus & all_clients_prev)
    total_row['Active Clients'] = total_active

    # Prepend TOTAL row
    total_df = pd.DataFrame([total_row])
    final_table = pd.concat([total_df, final_table], ignore_index=True)

    # Reorder columns
    col_order = ['Type of product'] + year_cols
    if growth_col:
        col_order.append(growth_col)
    col_order.extend(['No. of Clients', 'Active Clients'])
    final_table = final_table[col_order]

    # Styling
    def highlight_growth(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #166534; font-weight: bold'
            elif val < 0:
                return 'color: #dc2626; font-weight: bold'
        return ''

    def highlight_total_row(row):
        if row['Type of product'] == 'TOTAL':
            return ['background-color: #f0fdf4; color: #000000; font-weight: bold'] * len(row)
        return [''] * len(row)

    st.caption(f"Showing data for: {', '.join(map(str, years_sorted))}")

    format_dict = {col: "{:,.0f}" for col in year_cols}
    format_dict['No. of Clients'] = "{:,}"
    format_dict['Active Clients'] = "{:,}"
    if growth_col:
        format_dict[growth_col] = "{:+.1f}%"

    styled_table = final_table.style.format(format_dict).apply(highlight_total_row, axis=1)
    if growth_col:
        styled_table = styled_table.map(highlight_growth, subset=[growth_col])

    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Product Types", len(final_table) - 1)
    with col2:
        st.metric("Total Clients", total_row['No. of Clients'])
    with col3:
        st.metric("Active Clients", total_row['Active Clients'])
    with col4:
        if year_cols:
            total_vol = total_row[year_cols[-1]]
            st.metric(f"Volume ({years_sorted[-1]})", f"{total_vol:,.0f} KG")
    with col5:
        if growth_col:
            growth_val = total_row[growth_col]
            st.metric("Growth", f"{growth_val:+.1f}%", delta=f"{growth_val:+.1f}%")

    # =========================================================================
    # SECTION 2: Pie Chart by Type of Product (Top 10) with Channel Filter
    # =========================================================================
    st.divider()
    st.markdown("### 🥧 Distribution by Type of Product (Top 10)")

    # Channel filter
    if 'New Channel' in df_curr.columns:
        channels = sorted(df_curr['New Channel'].dropna().unique().tolist())
        selected_channel = st.selectbox(
            "Filter by Channel:",
            options=['All Channels'] + channels,
            key="type_growth_channel_filter"
        )

        # Apply channel filter
        if selected_channel == 'All Channels':
            df_pie = df_curr.copy()
        else:
            df_pie = df_curr[df_curr['New Channel'] == selected_channel]
    else:
        df_pie = df_curr.copy()
        st.info("Column 'New Channel' not found in data.")

    if df_pie.empty:
        st.warning("No data for selected channel.")
    else:
        # Aggregate by Type of product
        pie_data = df_pie.groupby('Type of product')['Sold'].sum().reset_index()
        pie_data = pie_data.sort_values('Sold', ascending=False)

        # Top 10 + Others
        top_10 = pie_data.head(10).copy()
        others_sum = pie_data.iloc[10:]['Sold'].sum() if len(pie_data) > 10 else 0

        if others_sum > 0:
            others_row = pd.DataFrame([{'Type of product': 'Others', 'Sold': others_sum}])
            top_10 = pd.concat([top_10, others_row], ignore_index=True)

        # Calculate percentage
        total_sold = top_10['Sold'].sum()
        top_10['Percentage'] = (top_10['Sold'] / total_sold * 100).round(1)
        top_10['Label'] = top_10['Type of product'] + '<br>' + top_10['Sold'].apply(lambda x: f"{x:,.0f} KG") + ' (' + top_10['Percentage'].astype(str) + '%)'

        # Pie chart with improved styling
        colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1

        fig_pie = px.pie(
            top_10,
            values='Sold',
            names='Type of product',
            title=f"<b>Sold Quantity by Type of Product</b><br><sup>{selected_channel}</sup>",
            hole=0.4,
            color_discrete_sequence=colors
        )

        # Better text formatting
        fig_pie.update_traces(
            textposition='outside',
            textinfo='label+percent+value',
            texttemplate='<b>%{label}</b><br>%{value:,.0f} KG<br>(%{percent})',
            textfont_size=11,
            pull=[0.02] * len(top_10),  # Slight pull for emphasis
            marker=dict(line=dict(color='#ffffff', width=2))
        )

        # Add total in center
        fig_pie.add_annotation(
            text=f"<b>Total</b><br>{total_sold:,.0f}<br>KG",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )

        fig_pie.update_layout(
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)"
            ),
            height=600,
            margin=dict(t=80, b=40, l=80, r=150),
            title=dict(
                font=dict(size=16),
                x=0.5,
                xanchor='center'
            )
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # =========================================================================
    # SECTION 3: Monthly Line Chart (Continuous Year-Month Axis)
    # =========================================================================
    st.divider()
    st.markdown("### 📈 Monthly Trend by Type of Product")

    # Get top product types for line chart
    top_types = df_curr.groupby('Type of product')['Sold'].sum().sort_values(ascending=False).head(10).index.tolist()

    selected_types = st.multiselect(
        "Select Product Types:",
        options=sorted(df_curr['Type of product'].unique()),
        default=top_types[:5],
        key="type_growth_line_filter"
    )

    if not selected_types:
        st.info("Please select at least one product type.")
    else:
        # Filter data
        df_line = df_curr[df_curr['Type of product'].isin(selected_types)].copy()

        # Create Year-Month column for continuous axis
        df_line['YearMonth'] = df_line['Year'].astype(int).astype(str) + '-' + df_line['Month'].astype(int).astype(str).str.zfill(2)
        df_line['Month_Int'] = df_line['Month'].astype(int)
        df_line['Year_Int'] = df_line['Year'].astype(int)

        # Aggregate
        monthly_data = df_line.groupby(['YearMonth', 'Year_Int', 'Month_Int', 'Type of product'])['Sold'].sum().reset_index()
        monthly_data = monthly_data.sort_values('YearMonth')

        # Calculate optimal Y-axis range based on selected data
        y_min = monthly_data['Sold'].min()
        y_max = monthly_data['Sold'].max()
        y_range_span = y_max - y_min

        # Add padding (20% above and below, or minimum padding if range is small)
        if y_range_span > 0:
            padding = max(y_range_span * 0.2, y_max * 0.1)
        else:
            padding = y_max * 0.2 if y_max > 0 else 100

        # Set y_min to 0 if data starts near 0, otherwise adjust
        if y_min < y_max * 0.3:
            y_axis_min = 0
        else:
            y_axis_min = max(0, y_min - padding)

        y_axis_max = y_max + padding

        # Format data labels
        monthly_data['DataLabel'] = monthly_data['Sold'].apply(lambda x: f"{x/1000:,.0f}K" if x >= 1000 else f"{x:,.0f}")

        # Line chart with data labels
        fig_line = px.line(
            monthly_data,
            x='YearMonth',
            y='Sold',
            color='Type of product',
            markers=True,
            text='DataLabel',
            title="Monthly Sold Quantity (KG) by Type of Product",
            labels={'YearMonth': 'Month', 'Sold': 'Sold (KG)', 'Type of product': 'Product Type'}
        )

        # Show data labels on points
        fig_line.update_traces(
            textposition='top center',
            textfont_size=9,
            hovertemplate='%{y:,.0f} KG'
        )

        # Build x-axis labels: T1, T2, ... T12 with year shown once per year
        unique_ym = list(monthly_data['YearMonth'].unique())
        tick_labels = []
        year_annotations = []
        year_dividers = []  # Vertical lines between years
        prev_year = None
        year_start_indices = {}  # Track where each year starts

        for i, ym in enumerate(unique_ym):
            year, month = ym.split('-')
            month_int = int(month)
            year_int = int(year)

            # Label as T1, T2, etc.
            tick_labels.append(f"T{month_int}")

            # Track year start index for divider positioning
            if year_int not in year_start_indices:
                year_start_indices[year_int] = i

            # Add year annotation at middle of each year's months
            if year_int != prev_year:
                year_months = [y for y in unique_ym if y.startswith(year)]
                mid_idx = len(year_months) // 2
                mid_ym = year_months[mid_idx] if mid_idx < len(year_months) else year_months[0]

                year_annotations.append(dict(
                    x=mid_ym,
                    y=-0.12,
                    xref='x',
                    yref='paper',
                    text=f"<b>{year}</b>",
                    showarrow=False,
                    font=dict(size=12, color='#333')
                ))

                # Add vertical divider before this year (except first year)
                if prev_year is not None and len(unique_ym) > 1:
                    # Calculate position as fraction of total width
                    divider_pos = (i - 0.5) / (len(unique_ym) - 1) if len(unique_ym) > 1 else 0.5
                    # Adjust for plot margins
                    divider_pos = max(0.01, min(0.99, divider_pos))
                    year_dividers.append(dict(
                        type='line',
                        x0=divider_pos,
                        x1=divider_pos,
                        y0=0,
                        y1=1,
                        xref='paper',
                        yref='paper',
                        line=dict(color='#999', width=1.5, dash='dash')
                    ))

                prev_year = year_int

        # --- Find max/min months per year and add star markers ---
        # Only show stars when exactly 1 product type is selected
        star_annotations = []

        if len(selected_types) == 1:
            # When 1 product type selected, show max/min stars
            monthly_totals = monthly_data.groupby(['YearMonth', 'Year_Int'])['Sold'].sum().reset_index()

            for year_val in monthly_totals['Year_Int'].unique():
                year_data = monthly_totals[monthly_totals['Year_Int'] == year_val]
                if len(year_data) > 1:  # Need at least 2 months to compare
                    max_row = year_data.loc[year_data['Sold'].idxmax()]
                    min_row = year_data.loc[year_data['Sold'].idxmin()]

                    # Green star for max month (using HTML)
                    star_annotations.append(dict(
                        x=max_row['YearMonth'],
                        y=max_row['Sold'],
                        xref='x',
                        yref='y',
                        text='<b>★</b>',
                        showarrow=False,
                        font=dict(size=20, color='#22c55e'),
                        yshift=25
                    ))

                    # Red star for min month (using HTML)
                    star_annotations.append(dict(
                        x=min_row['YearMonth'],
                        y=min_row['Sold'],
                        xref='x',
                        yref='y',
                        text='<b>★</b>',
                        showarrow=False,
                        font=dict(size=20, color='#ef4444'),
                        yshift=-30
                    ))

        # Combine all annotations
        all_annotations = year_annotations + star_annotations

        # Update layout
        fig_line.update_layout(
            template="plotly_white",
            xaxis=dict(
                tickmode='array',
                tickvals=unique_ym,
                ticktext=tick_labels,
                tickangle=0,
                title=''
            ),
            yaxis=dict(
                range=[y_axis_min, y_axis_max],
                fixedrange=False,
                title='Sold (KG)',
                tickformat=',d'
            ),
            annotations=all_annotations,
            shapes=year_dividers,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                itemclick='toggle',
                itemdoubleclick='toggleothers'
            ),
            height=550,
            hovermode='x unified',
            margin=dict(b=100),
            uirevision='constant'  # Preserve UI state
        )

        # Add legend note for stars (only when 1 product selected)
        if len(selected_types) == 1:
            st.caption("★ Xanh = Tháng cao nhất trong năm | ★ Đỏ = Tháng thấp nhất trong năm")

        # Help text with autoscale instructions
        with st.expander("💡 Hướng dẫn sử dụng chart", expanded=False):
            st.markdown("""
            - **Click legend**: Toggle on/off product type
            - **Double-click legend**: Chỉ hiển thị 1 product type
            - **🔄 Auto-scale Y**: Sau khi toggle, click nút **Autoscale** (icon 📐) trên toolbar hoặc **double-click** vào vùng chart
            - **Zoom**: Kéo chuột để zoom vùng cần xem
            - **Reset**: Click nút **Reset axes** hoặc double-click vào chart
            """)

        # Enable auto-rescale on legend click
        st.plotly_chart(
            fig_line,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'modeBarButtonsToAdd': ['autoScale2d', 'resetScale2d'],
                'scrollZoom': True,
                'doubleClick': 'autosize'  # Double-click to auto-scale
            }
        )

        # Show data table
        with st.expander("View Monthly Data"):
            pivot_monthly = monthly_data.pivot(
                index='YearMonth',
                columns='Type of product',
                values='Sold'
            ).fillna(0)
            pivot_monthly['Total'] = pivot_monthly.sum(axis=1)
            pivot_monthly = pivot_monthly.sort_index()

            st.dataframe(
                pivot_monthly.style.format("{:,.0f}"),
                use_container_width=True
            )

    # =========================================================================
    # SECTION 4: Channel Growth % by Year
    # =========================================================================
    st.divider()
    st.markdown("### 📊 Channel Growth Analysis")

    # Check if 'New Channel' column exists
    if 'New Channel' not in df.columns:
        st.info("Column 'New Channel' not found in data. Channel Growth Analysis requires this column.")
    # Need at least 2 years for comparison
    elif len(years_sorted) >= 2:
        # Step 1: Calculate % growth for each channel by year
        # Aggregate sold quantity by Channel and Year
        channel_by_year = df.groupby(['New Channel', 'Year'])['Sold'].sum().reset_index()
        channel_by_year = channel_by_year[channel_by_year['Year'].isin(years_sorted)]

        # Pivot: Channel as rows, Year as columns
        channel_pivot = channel_by_year.pivot(
            index='New Channel',
            columns='Year',
            values='Sold'
        ).fillna(0)

        # Calculate YoY growth for each consecutive year pair
        growth_data = []
        year_cols = sorted([col for col in channel_pivot.columns])

        for i in range(1, len(year_cols)):
            prev_year = year_cols[i - 1]
            curr_year = year_cols[i]
            growth_label = f"{int(curr_year)} vs {int(prev_year)}"

            for channel in channel_pivot.index:
                prev_val = channel_pivot.loc[channel, prev_year]
                curr_val = channel_pivot.loc[channel, curr_year]

                if prev_val > 0:
                    growth_pct = ((curr_val - prev_val) / prev_val) * 100
                else:
                    growth_pct = 100 if curr_val > 0 else 0

                growth_data.append({
                    'Channel': channel,
                    'Period': growth_label,
                    'Growth %': growth_pct,
                    'Prev Value': prev_val,
                    'Curr Value': curr_val
                })

        growth_df = pd.DataFrame(growth_data)

        # Step 2: Create clustered column chart
        st.markdown(f"#### 📈 % Growth Sold Quantity by Channel (Year over Year)")

        # Sort channels by latest growth
        latest_period = growth_df['Period'].unique()[-1]
        channel_order = growth_df[growth_df['Period'] == latest_period].sort_values('Growth %', ascending=True)['Channel'].tolist()

        # Create chart
        fig_growth = px.bar(
            growth_df,
            x='Growth %',
            y='Channel',
            color='Period',
            barmode='group',
            orientation='h',
            title="% Growth Sold Quantity by Channel",
            labels={'Growth %': 'Growth (%)', 'Channel': 'Channel'},
            color_discrete_sequence=px.colors.qualitative.Set2,
            category_orders={'Channel': channel_order}
        )

        # Add reference line at 0
        fig_growth.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

        # Color bars based on positive/negative
        fig_growth.update_traces(
            texttemplate='%{x:+.0f}%',
            textposition='outside',
            textfont_size=9
        )

        fig_growth.update_layout(
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                title=""
            ),
            margin=dict(l=20, r=100, t=60, b=20),
            xaxis=dict(
                ticksuffix='%',
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=1
            )
        )

        st.plotly_chart(fig_growth, use_container_width=True)

        # Summary table
        with st.expander("📋 View Detailed Growth Data"):
            # Pivot for display
            display_pivot = growth_df.pivot(
                index='Channel',
                columns='Period',
                values='Growth %'
            ).reset_index()

            # Sort by latest period
            display_pivot = display_pivot.sort_values(display_pivot.columns[-1], ascending=False)

            # Format as percentage
            format_dict = {col: "{:+.1f}%" for col in display_pivot.columns if col != 'Channel'}

            # Style with color based on positive/negative values
            def color_growth(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: #166534; background-color: #dcfce7'
                    elif val < 0:
                        return 'color: #dc2626; background-color: #fee2e2'
                return ''

            growth_cols = [col for col in display_pivot.columns if col != 'Channel']
            st.dataframe(
                display_pivot.style.format(format_dict).map(color_growth, subset=growth_cols),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.info("Chọn ít nhất 2 năm để xem so sánh Channel Growth.")

