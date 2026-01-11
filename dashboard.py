r"""
Sales Dashboard Project
Single-file distribution containing helper functions, KPI generation, Plotly charts, Streamlit app, and AI insight generator.

HOW TO USE (ENGLISH):
1. Create a virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

2. Place your Excel file next to this script and set FILE_PATH variable or pass as CLI arg.
   If Excel is too large, save as CSV and use that instead (faster for pandas).

3. Run locally:
   # Quick KPI run (prints & saves csv reports)
   python dashboard.py --file sales.xlsx --mode kpis

   # Run Streamlit dashboard
   streamlit run dashboard.py -- --file sales.xlsx --mode streamlit
"""

import argparse
import sys
import datetime as dt
import streamlit as st
import pandas as pd

# Import modules
from src.data_processing import load_data, clean_data
from src.analysis import (
    compute_top_level_kpis,
    compute_client_metrics,
    compute_product_metrics,
    compute_region_metrics,
    compute_rfm_clusters
)
from src.utils import (
    ai_insights_summary,
    export_reports,
    filter_by_date
)
from src.ui_helpers import apply_custom_styles, checkbox_filter

# Import Tab Renderers
from src.tabs.executive_overview import render_executive_overview
from src.tabs.product_intelligence import render_product_intelligence
from src.tabs.customer_market import render_customer_market
from src.tabs.growth_insights import render_growth_insights
from src.tabs.vietnam_focus import render_vietnam_focus
from src.tabs.product_launching import render_product_launching
from src.tabs.product_type_growth import render_product_type_growth

# -------------------------
# Streamlit App
# -------------------------

def streamlit_app(df: pd.DataFrame = None):
    st.set_page_config(layout='wide', page_title='Professional Sales Dashboard')

    # --- CSS: Professional Theme ---
    apply_custom_styles()

    # If no dataframe passed (e.g. not loaded via CLI arg), show uploader
    raw_cols = []
    if df is None:
        st.info("No file provided via CLI. Please upload your Sales Data (Excel or CSV).")

        # Clear cache button
        if st.button("🔄 Clear Cache & Reload"):
            st.cache_data.clear()
            st.rerun()

        uploaded_file = st.file_uploader("Upload Sales Data", type=['xlsx', 'xls', 'csv'])
        if uploaded_file is not None:
            try:
                # Clear cache when new file uploaded to ensure fresh data
                st.cache_data.clear()

                # Load Raw
                df_raw = load_data(uploaded_file)
                raw_cols = list(df_raw.columns)

                # Check for Multiple Sheets? read_excel reads 1st by default.

                # Clean
                df = clean_data(df_raw)
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
        else:
            st.stop()
    else:
        # CLI usage, we don't have raw cols easily unless we change main(). 
        pass

    # --- Sidebar: Global Controls ---
    st.sidebar.title("📊 Control Panel")
    
    # Debug Info
    with st.sidebar.expander("🛠 Data Debug Info"):
        st.write(f"**Rows loaded (Cleaned):** {len(df)}")
        st.write(f"**Total Columns:** {len(df.columns)}")
        st.write(f"**Cleaned Columns:** {list(df.columns)}")
        st.write("**Unique Years:**", df['Year'].unique())
        st.write("**Unique Regions:**", df['Region'].unique())
        # Check for New Channel column
        if 'New Channel' in df.columns:
            st.success(f"✅ 'New Channel' column found! Values: {df['New Channel'].nunique()} unique")
            st.write("Sample values:", df['New Channel'].dropna().unique()[:10].tolist())
        else:
            st.error("❌ 'New Channel' column NOT found!")
            # Show columns containing 'channel' (case-insensitive)
            channel_cols = [c for c in df.columns if 'channel' in c.lower()]
            if channel_cols:
                st.write("Similar columns found:", channel_cols)
        st.write("First 3 rows (Cleaned):")
        st.dataframe(df.head(3))
    
    # Year Filter (Checkbox)
    years = sorted(df['Year'].dropna().astype(int).unique())
    if not years:
        st.error("No valid Years found in data!")
        st.stop()
        
    # Default to Current System Year if available, else latest year
    current_year_sys = dt.datetime.now().year
    default_year_list = [current_year_sys] if current_year_sys in years else ([years[-1]] if years else [])
    
    # User Request: Default expand Years and Regions
    selected_years = checkbox_filter('Select Years', years, key_prefix="year_filter", default_selected=default_year_list, expanded=True)
    if not selected_years:
        st.warning("Please select at least one year.")
        st.stop()
    
    # Month Filter (Checkbox)
    all_months = sorted(df['Month'].dropna().unique())
    selected_months = checkbox_filter('Select Months', all_months, key_prefix="month_filter")
    
    if not selected_months:
         st.warning("Please select at least one month.")
         st.stop() 
    
    # Region Filter (Checkbox)
    unique_regions = sorted(df['Region'].dropna().unique().tolist())
    selected_regions = checkbox_filter('Select Regions', unique_regions, key_prefix="region_filter", expanded=True)
    if not selected_regions:
         st.warning("Please select at least one region.")
         st.stop()
         
    # Channel Filter (Checkbox)
    selected_channels = []
    if 'New Channel' in df.columns:
        unique_channels = sorted(df['New Channel'].dropna().unique().tolist())
        selected_channels = checkbox_filter('Select Channels', unique_channels, key_prefix="channel_filter")
    else:
        st.sidebar.info("ℹ️ Column 'New Channel' not found")
        
    # Country Filter (Checkbox)
    selected_countries = []
    if 'Country' in df.columns:
        unique_countries = sorted(df['Country'].dropna().unique().tolist())
        selected_countries = checkbox_filter('Select Country', unique_countries, key_prefix="country_filter")
    
    # --- Data Filtering ---
    # 1. Current Period Data
    df_curr = filter_by_date(df, selected_years, selected_months)
    df_curr = df_curr[df_curr['Region'].isin(selected_regions)]
    
    # Apply Channel Filter
    if 'New Channel' in df.columns:
        if selected_channels:
            df_curr = df_curr[df_curr['New Channel'].isin(selected_channels)]
        else:
            st.warning("Please select at least one Channel.")
            st.stop()
        
    # Apply Country Filter
    if 'Country' in df.columns and selected_countries:
        df_curr = df_curr[df_curr['Country'].isin(selected_countries)]
    elif 'Country' in df.columns and not selected_countries:
        st.warning("Please select at least one Country.")
        st.stop()

    # 2. Comparison Logic (Dynamic)
    # Only enable comparison if EXACTLY ONE year is selected, to avoid ambiguity.
    single_year_mode = (len(selected_years) == 1)
    current_year_val = selected_years[0] if single_year_mode else None
    
    has_prev_year = False
    df_prev = pd.DataFrame()

    if single_year_mode:
        has_prev_year = (current_year_val - 1) in years
        
        if has_prev_year:
            # Standard YoY
            df_prev = filter_by_date(df, [current_year_val - 1], selected_months)
            df_prev = df_prev[df_prev['Region'].isin(selected_regions)]
            
            # Apply Channel Filter
            if 'New Channel' in df.columns and selected_channels:
                df_prev = df_prev[df_prev['New Channel'].isin(selected_channels)]
                
            # Apply Country Filter
            if 'Country' in df.columns and selected_countries:
                df_prev = df_prev[df_prev['Country'].isin(selected_countries)]
    else:
        # Fallback: MoM (Month-over-Month) or Multi-Year View
        pass

    # --- Main UI ---
    title_text = f"🚀 Business Performance: {', '.join(map(str, selected_years))}" if len(selected_years) <= 3 else f"🚀 Business Performance: {len(selected_years)} Years Selected"
    st.title(title_text)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Executive Overview", "📦 Product Intelligence", "👥 Customer & Market", "📈 Growth & Insights", "🇻🇳 Vietnam Focus", "🚀 Product Launching", "📈 Type Growth"])

    # === TAB 1: EXECUTIVE OVERVIEW ===
    with tab1:
        render_executive_overview(df_curr, df_prev, selected_years, selected_months, current_year_val, single_year_mode, has_prev_year)

    # === TAB 2: PRODUCT INTELLIGENCE ===
    with tab2:
        render_product_intelligence(df_curr, selected_years, df=df)

    # === TAB 3: CUSTOMER & MARKET ===
    with tab3:
        render_customer_market(df, df_curr)

    # === TAB 4: GROWTH & INSIGHTS ===
    with tab4:
        render_growth_insights(df_curr, df_prev, selected_years, selected_months, current_year_val, single_year_mode, has_prev_year, df, selected_regions)

    # === TAB 5: LOCAL VIETNAM FOCUS ===
    with tab5:
        render_vietnam_focus(df, df_curr, df_prev, has_prev_year, current_year_val)

    # === TAB 6: PRODUCT LAUNCHING ===
    with tab6:
        render_product_launching(df, df_curr, df_prev, current_year_val)

    # === TAB 7: PRODUCT TYPE GROWTH ===
    with tab7:
        render_product_type_growth(df, df_curr, selected_years)


# -------------------------
# CLI Entrypoint
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    # Make file optional
    parser.add_argument('--file', type=str, default=None, help='Path to sales.xlsx or sales.csv')
    parser.add_argument('--mode', type=str, default='kpis', choices=['kpis', 'streamlit'], help='mode')
    parser.add_argument('--out', type=str, default='reports_out', help='output folder')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI key for AI insights (optional)')
    args, unknown = parser.parse_known_args()

    # Smart default: If no file provided, assume streamlit mode (interactive)
    # This fixes usage: `streamlit run dashboard.py` without args
    if args.file is None and args.mode == 'kpis':
        # Check if user explicitly passed --mode kpis
        if '--mode' not in sys.argv and '-m' not in sys.argv:
             args.mode = 'streamlit'

    # If mode is explicitly kpis, file is required
    if args.mode == 'kpis' and not args.file:
        print("Error: --file argument is required in 'kpis' mode.")
        sys.exit(1)

    df = None
    if args.file:
        df = load_data(args.file)
        df = clean_data(df)

    if args.mode == 'kpis':
        kpis = compute_top_level_kpis(df)
        client_df = compute_client_metrics(df)
        prod_df = compute_product_metrics(df)
        region_df = compute_region_metrics(df)

        # clustering
        client_df = compute_rfm_clusters(client_df)

        # export
        export_reports(args.out, kpis, client_df, prod_df, region_df)

        # print short summary
        print('\n-- TOP KPIS --')
        print(f"Total revenue: {kpis['total_revenue']:,}")
        print(f"Total KG: {kpis['total_kg']:,}")
        if kpis.get('yoy_growth') is not None:
            print(f"YoY growth: {kpis['yoy_growth']:.2%}")
        if args.openai_key:
            print('\nAI Insights:')
            print(ai_insights_summary(kpis, client_df, prod_df, openai_api_key=args.openai_key))

    elif args.mode == 'streamlit':
        # Hand off to streamlit; when streamlit runs it will import and call this script
        streamlit_app(df)


if __name__ == '__main__':
    main()
