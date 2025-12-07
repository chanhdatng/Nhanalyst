import pandas as pd
import datetime as dt

# Load data utilizing the same logic as dashboard.py (simplified)
def load_and_clean():
    try:
        df = pd.read_excel('data.xlsx', engine='openpyxl')
        
        # Renaissance of clean_data logic
        df['Quantity (KG)'] = pd.to_numeric(df['Quantity (KG)'], errors='coerce').fillna(0.0)
        df['Sold'] = df['Quantity (KG)'] # Logic from dashboard.py

        # Date Logic
        # Assuming Year/Month columns exist
        def make_date(r):
            try:
                y = int(r['Year'])
                m = r['Month']
                if isinstance(m, str):
                    import re
                    digits = re.findall(r'\d+', m)
                    m = int(digits[0]) if digits else 1
                return dt.date(y, m, 1)
            except:
                return pd.NaT

        df['date__ym'] = df.apply(make_date, axis=1)
        df['date__ym'] = pd.to_datetime(df['date__ym'])
        df = df[~df['date__ym'].isna()]
        return df
    except Exception as e:
        print(f"Error loading: {e}")
        return pd.DataFrame()

df = load_and_clean()

if not df.empty:
    max_date = df['date__ym'].max()
    print(f"Max Date in Data: {max_date}")
    
    cutoff_6m = max_date - pd.DateOffset(months=6)
    print(f"Cutoff 6 months: {cutoff_6m}")
    
    # Check distinct clients in last 6 months
    df_6m = df[df['date__ym'] >= cutoff_6m]
    print(f"Rows in last 6 months: {len(df_6m)}")
    
    # Group by Product + Client
    print("\n--- Sample Active Analysis ---")
    grp = df_6m.groupby(['Name of product', 'Name of client']).size().reset_index(name='OrderCount')
    
    # Filter >= 2
    active = grp[grp['OrderCount'] >= 2]
    
    print(f"Total pairs (Product, Client) in last 6m: {len(grp)}")
    print(f"Active pairs (>=2 orders): {len(active)}")
    
    if len(active) > 0:
        print("Sample Active Calculation:")
        print(active.head(5))
        
    # Check specific case if possible. Look for a product with high volume but 0 active?
    # We can just list top 5 products by row count in last 6m
    print("\nTop 5 Products by Activity (Row Count in last 6m):")
    top_prod = df_6m['Name of product'].value_counts().head(5)
    print(top_prod)
    
else:
    print("DataFrame empty.")
