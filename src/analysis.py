import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from src.utils import DEFAULT_DATE_COL, calculate_growth

def compute_top_level_kpis(df: pd.DataFrame) -> dict:
    """Compute the top-level KPIs described in the design.
    Returns a dict of simple scalars and series.
    """
    kpis = {}
    total_revenue = df['Sold'].sum()
    total_kg = df['Quantity (KG)'].sum()
    active_clients = df['Name of client'].nunique()

    # average revenue per client
    rev_per_client = df.groupby('Name of client')['Sold'].sum().mean()

    # Monthly aggregated series for growth calculations
    monthly = df.groupby(pd.Grouper(key=DEFAULT_DATE_COL, freq='M')).agg({'Sold': 'sum'}).sort_index()

    # YoY growth (last full year vs previous)
    years = df['Year'].dropna().unique()
    years = sorted([int(y) for y in years if not pd.isna(y)])
    yoy = None
    if len(years) >= 2:
        last_year = years[-1]
        prev_year = years[-2]
        last_sum = df.loc[df['Year'] == last_year, 'Sold'].sum()
        prev_sum = df.loc[df['Year'] == prev_year, 'Sold'].sum()
        yoy = (last_sum - prev_sum) / prev_sum if prev_sum != 0 else np.nan

    # MoM growth (compare last month vs previous month)
    mom = None
    if len(monthly) >= 2:
        mom = (monthly['Sold'].iloc[-1] - monthly['Sold'].iloc[-2]) / (monthly['Sold'].iloc[-2] if monthly['Sold'].iloc[-2] != 0 else np.nan)

    # New / churned clients
    last_date = df[DEFAULT_DATE_COL].max()
    cutoff_new = last_date - pd.DateOffset(months=12)
    clients_last_year = set(df.loc[df[DEFAULT_DATE_COL] >= cutoff_new, 'Name of client'].unique())
    all_clients = set(df['Name of client'].unique())
    new_clients = clients_last_year  # naive: clients who appear in last 12 months (could refine)

    # churn: clients with no orders in last 3 months
    cutoff_churn = last_date - pd.DateOffset(months=3)
    active_recent = set(df.loc[df[DEFAULT_DATE_COL] >= cutoff_churn, 'Name of client'].unique())
    churned_clients = list(all_clients - active_recent)

    # Top product / fruit
    top_product = df.groupby('Name of product')['Sold'].sum().sort_values(ascending=False).head(1)
    top_fruit = df.groupby('Kind of fruit')['Sold'].sum().sort_values(ascending=False).head(1)

    kpis.update({
        'total_revenue': float(total_revenue),
        'total_kg': float(total_kg),
        'active_clients': int(active_clients),
        'avg_revenue_per_client': float(rev_per_client),
        'monthly_series': monthly.reset_index(),
        'yoy_growth': float(yoy) if yoy is not None else None,
        'mom_growth': float(mom) if mom is not None else None,
        'new_clients_last_12m_count': len(new_clients),
        'churned_clients_count': len(churned_clients),
        'top_product': dict(top_product.head(1)) if not top_product.empty else {},
        'top_fruit': dict(top_fruit.head(1)) if not top_fruit.empty else {}
    })

    return kpis


def compute_client_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe aggregated by client with key metrics: revenue, kg, orders, freq, recency, LTV (simple), RFM score.
    """
    now = df[DEFAULT_DATE_COL].max() + pd.DateOffset(days=1)
    client = df.groupby('Name of client').agg(
        revenue=('Sold', 'sum'),
        kg=('Quantity (KG)', 'sum'),
        orders=('Sold', 'count'),
        last_order=(DEFAULT_DATE_COL, 'max'),
        first_order=(DEFAULT_DATE_COL, 'min')
    ).reset_index()

    client['recency_days'] = (now - client['last_order']).dt.days
    client['frequency'] = client['orders']
    client['monetary'] = client['revenue']

    # Simple RFM scoring: quantiles
    client['r_score'] = pd.qcut(client['recency_days'].rank(method='first'), 5, labels=range(5, 0, -1)).astype(int)
    client['f_score'] = pd.qcut(client['frequency'].rank(method='first'), 5, labels=range(1, 6)).astype(int)
    client['m_score'] = pd.qcut(client['monetary'].rank(method='first'), 5, labels=range(1, 6)).astype(int)
    client['rfm_score'] = client['r_score']*100 + client['f_score']*10 + client['m_score']

    return client.sort_values('revenue', ascending=False)


def compute_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    prod = df.groupby(['Name of product', 'SKU', 'Kind of fruit', 'Type of product']).agg(
        revenue=('Sold', 'sum'),
        kg=('Quantity (KG)', 'sum'),
        orders=('Sold', 'count')
    ).reset_index()
    prod['price_per_kg'] = prod['revenue'] / prod['kg'].replace({0: np.nan})
    prod = prod.sort_values('revenue', ascending=False)
    return prod


def compute_region_metrics(df: pd.DataFrame) -> pd.DataFrame:
    r = df.groupby(['Region', 'Country']).agg(
        revenue=('Sold', 'sum'),
        kg=('Quantity (KG)', 'sum')
    ).reset_index().sort_values('revenue', ascending=False)
    return r


def compute_rfm_clusters(client_df: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
    features = client_df[['recency_days', 'frequency', 'monetary']].fillna(0)
    # Simple scaling (min-max)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    client_df['cluster'] = kmeans.fit_predict(X)
    return client_df


# =============================================================================
# PHASE 1: CORE ANALYSIS FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def compute_financial_health_score(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    """
    Compute financial health score (0-100) from 4 weighted components.
    
    Note: 'Sold' column represents Volume (KG), not monetary revenue.

    Args:
        df_curr: Current period data
        df_prev: Previous period data (can be empty)

    Returns:
        dict: {
            'score': float (0-100),
            'color': str ('red'/'yellow'/'green'),
            'components': dict with component breakdowns
        }
    """
    # Edge case: empty df_prev
    if df_prev is None or df_prev.empty:
        df_prev = pd.DataFrame(columns=df_curr.columns)
    
    # Edge case: empty df_curr
    if df_curr is None or df_curr.empty:
        return {
            'score': 0,
            'color': 'red',
            'components': {
                'volume_growth': {'value': 0, 'score': 0, 'weight': 0.30},
                'avg_order_size': {'value': 0, 'score': 0, 'weight': 0.25},
                'retention': {'value': 0, 'score': 0, 'weight': 0.25},
                'order_size_growth': {'value': 0, 'score': 0, 'weight': 0.20}
            }
        }
    
    # Component 1: Volume Growth Score (30% weight)
    volume_curr = df_curr['Sold'].sum()  # Volume in KG
    volume_prev = df_prev['Sold'].sum() if not df_prev.empty else 0
    
    growth_raw = calculate_growth(volume_curr, volume_prev)
    growth_pct = growth_raw * 100 if growth_raw is not None else 0
    
    # Scoring thresholds for volume growth
    if growth_pct >= 20:
        volume_growth_score = 100
    elif growth_pct >= 10:
        volume_growth_score = 50 + (growth_pct - 10) * 5  # Linear interpolation
    elif growth_pct >= 0:
        volume_growth_score = 25 + growth_pct * 2.5
    else:
        volume_growth_score = max(0, 25 + growth_pct)
    
    # Component 2: Avg Order Size (25% weight) - Volume per order in KG
    avg_order_size_curr = volume_curr / len(df_curr) if len(df_curr) > 0 else 0
    # Normalize to 0-100 (assume 2000 KG = excellent avg order)
    avg_order_size_score = min(100, (avg_order_size_curr / 2000) * 100)
    avg_order_size_value = avg_order_size_curr
    
    # Component 3: Customer Retention (25% weight)
    clients_curr = set(df_curr['Name of client'].unique())
    clients_prev = set(df_prev['Name of client'].unique()) if not df_prev.empty else set()
    
    if clients_prev:
        retained = len(clients_curr & clients_prev)
        retention_rate = (retained / len(clients_prev)) * 100
    else:
        retention_rate = 100  # No previous data = assume 100%
    
    retention_score = min(100, retention_rate)
    
    # Component 4: Avg Order Size Growth (20% weight)
    avg_order_size_prev = volume_prev / len(df_prev) if len(df_prev) > 0 else 0
    order_size_growth_raw = calculate_growth(avg_order_size_curr, avg_order_size_prev)
    order_size_growth_pct = order_size_growth_raw * 100 if order_size_growth_raw is not None else 0
    
    if order_size_growth_pct >= 15:
        order_size_growth_score = 100
    elif order_size_growth_pct >= 5:
        order_size_growth_score = 50 + (order_size_growth_pct - 5) * 5
    elif order_size_growth_pct >= 0:
        order_size_growth_score = 25 + order_size_growth_pct * 5
    else:
        order_size_growth_score = max(0, 25 + order_size_growth_pct * 2)
    
    # Final Score (weighted)
    final_score = (
        volume_growth_score * 0.30 +
        avg_order_size_score * 0.25 +
        retention_score * 0.25 +
        order_size_growth_score * 0.20
    )
    
    # Color determination
    if final_score >= 75:
        color = 'green'
    elif final_score >= 50:
        color = 'yellow'
    else:
        color = 'red'
    
    return {
        'score': round(final_score, 1),
        'color': color,
        'components': {
            'volume_growth': {'value': round(growth_pct, 1), 'score': round(volume_growth_score, 1), 'weight': 0.30},
            'avg_order_size': {'value': round(avg_order_size_value, 2), 'score': round(avg_order_size_score, 1), 'weight': 0.25},
            'retention': {'value': round(retention_rate, 1), 'score': round(retention_score, 1), 'weight': 0.25},
            'order_size_growth': {'value': round(order_size_growth_pct, 1), 'score': round(order_size_growth_score, 1), 'weight': 0.20}
        }
    }



@st.cache_data(ttl=3600)
def compute_churn_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-customer churn risk scores.

    Args:
        df: Full historical data with date__ym, Sold, Name of client, Name of product columns

    Returns:
        DataFrame columns: ['Name of client', 'churn_risk_score', 'risk_level',
                           'days_since_last', 'frequency_trend', 'volume_trend',
                           'variety_trend', 'total_volume']
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'Name of client', 'churn_risk_score', 'risk_level',
            'days_since_last', 'frequency_trend', 'volume_trend',
            'variety_trend', 'total_volume'
        ])
    
    today = df[DEFAULT_DATE_COL].max()
    
    # Vectorized approach using groupby operations
    # Get per-client stats
    client_last_purchase = df.groupby('Name of client')[DEFAULT_DATE_COL].max()
    client_total_volume = df.groupby('Name of client')['Sold'].sum()  # Volume in KG
    
    # Define time windows
    cutoff_90d = today - pd.Timedelta(days=90)
    cutoff_180d = today - pd.Timedelta(days=180)
    
    # Split data into time periods
    last_3m = df[df[DEFAULT_DATE_COL] >= cutoff_90d]
    prev_3m = df[(df[DEFAULT_DATE_COL] >= cutoff_180d) & (df[DEFAULT_DATE_COL] < cutoff_90d)]
    
    # Aggregate stats for each period
    last_3m_orders = last_3m.groupby('Name of client').size()
    prev_3m_orders = prev_3m.groupby('Name of client').size()
    
    last_3m_volume = last_3m.groupby('Name of client')['Sold'].sum()
    prev_3m_volume = prev_3m.groupby('Name of client')['Sold'].sum()
    
    last_3m_products = last_3m.groupby('Name of client')['Name of product'].nunique()
    prev_3m_products = prev_3m.groupby('Name of client')['Name of product'].nunique()
    
    client_stats = []
    
    for client in df['Name of client'].unique():
        # 1. Recency Score (40%)
        last_purchase = client_last_purchase.get(client, today)
        days_since = (today - last_purchase).days
        
        if days_since <= 30:
            recency_score = 0
        elif days_since <= 60:
            recency_score = (days_since - 30)
        elif days_since <= 90:
            recency_score = 30 + (days_since - 60)
        else:
            recency_score = min(100, 60 + (days_since - 90) * 0.5)
        
        # 2. Frequency Decline (30%)
        orders_recent = last_3m_orders.get(client, 0)
        orders_prev = prev_3m_orders.get(client, 0)
        
        if orders_prev > 0:
            freq_change = (orders_recent - orders_prev) / orders_prev
            frequency_score = max(0, min(100, (1 - freq_change) * 50))
        else:
            frequency_score = 50 if orders_recent == 0 else 0  # New customer with recent orders = low risk
        
        # 3. Volume Decline (20%)
        volume_recent = last_3m_volume.get(client, 0)
        volume_prev = prev_3m_volume.get(client, 0)
        
        if volume_prev > 0:
            volume_change = (volume_recent - volume_prev) / volume_prev
            volume_score = max(0, min(100, (1 - volume_change) * 50))
        else:
            volume_score = 50 if volume_recent == 0 else 0
        
        # 4. Variety Decline (10%)
        products_recent = last_3m_products.get(client, 0)
        products_prev = prev_3m_products.get(client, 0)
        
        if products_prev > 0:
            variety_change = (products_recent - products_prev) / products_prev
            variety_score = max(0, min(100, (1 - variety_change) * 50))
        else:
            variety_score = 50 if products_recent == 0 else 0
        
        # Final Churn Risk
        churn_risk = (
            recency_score * 0.40 +
            frequency_score * 0.30 +
            volume_score * 0.20 +
            variety_score * 0.10
        )
        
        risk_level = 'High' if churn_risk >= 70 else ('Medium' if churn_risk >= 40 else 'Low')
        
        client_stats.append({
            'Name of client': client,
            'churn_risk_score': round(churn_risk, 1),
            'risk_level': risk_level,
            'days_since_last': days_since,
            'frequency_trend': round(freq_change * 100, 1) if orders_prev > 0 else None,
            'volume_trend': round(volume_change * 100, 1) if volume_prev > 0 else None,
            'variety_trend': round(variety_change * 100, 1) if products_prev > 0 else None,
            'total_volume': client_total_volume.get(client, 0)
        })
    
    result_df = pd.DataFrame(client_stats)
    return result_df.sort_values('churn_risk_score', ascending=False).reset_index(drop=True)


@st.cache_data(ttl=3600)
def compute_product_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify products into lifecycle stages: Introduction, Growth, Maturity, Decline.
    
    Note: 'Sold' column represents Volume (KG), not monetary revenue.

    Args:
        df: Full historical data

    Returns:
        DataFrame columns: ['Name of product', 'lifecycle_stage', 'stage_emoji',
                           'age_months', 'growth_rate', 'total_volume', 'recent_volume']
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'Name of product', 'lifecycle_stage', 'stage_emoji',
            'age_months', 'growth_rate', 'total_volume', 'recent_volume'
        ])
    
    today = df[DEFAULT_DATE_COL].max()
    cutoff_90d = today - pd.Timedelta(days=90)
    cutoff_180d = today - pd.Timedelta(days=180)
    
    # Pre-compute aggregates for performance
    product_first_sale = df.groupby('Name of product')[DEFAULT_DATE_COL].min()
    product_total_volume = df.groupby('Name of product')['Sold'].sum()  # Volume in KG
    
    last_3m = df[df[DEFAULT_DATE_COL] >= cutoff_90d]
    prev_3m = df[(df[DEFAULT_DATE_COL] >= cutoff_180d) & (df[DEFAULT_DATE_COL] < cutoff_90d)]
    
    last_3m_volume = last_3m.groupby('Name of product')['Sold'].sum()
    prev_3m_volume = prev_3m.groupby('Name of product')['Sold'].sum()
    
    product_stages = []
    
    for product in df['Name of product'].unique():
        # Calculate age
        first_sale = product_first_sale.get(product, today)
        age_months = max(1, ((today - first_sale).days // 30))
        
        # Calculate volume
        total_volume = product_total_volume.get(product, 0)
        volume_recent = last_3m_volume.get(product, 0)
        volume_prev = prev_3m_volume.get(product, 0)
        
        # Calculate growth rate
        if volume_prev > 0:
            growth_rate = (volume_recent - volume_prev) / volume_prev
        else:
            growth_rate = 0 if volume_recent == 0 else 1
        
        # Lifecycle stage logic (thresholds adjusted for volume in KG)
        if age_months < 6 and total_volume < 10000 and growth_rate > 0.5:
            stage, emoji = 'Introduction', '🌱'
        elif age_months < 6 and growth_rate > 0:
            stage, emoji = 'Introduction', '🌱'
        elif age_months <= 18 and growth_rate > 0.2:
            stage, emoji = 'Growth', '📈'
        elif growth_rate < -0.2:
            stage, emoji = 'Decline', '📉'
        elif growth_rate < -0.1:
            # Check for consistent decline
            prod_df = df[df['Name of product'] == product]
            monthly = prod_df.groupby(prod_df[DEFAULT_DATE_COL].dt.to_period('M'))['Sold'].sum()
            if len(monthly) >= 3:
                recent_months = monthly.iloc[-3:]
                if len(recent_months) >= 2:
                    diffs = recent_months.diff().dropna()
                    if len(diffs) > 0 and (diffs < 0).all():
                        stage, emoji = 'Decline', '📉'
                    else:
                        stage, emoji = 'Maturity', '💎'
                else:
                    stage, emoji = 'Maturity', '💎'
            else:
                stage, emoji = 'Maturity', '💎'
        else:
            stage, emoji = 'Maturity', '💎'
        
        product_stages.append({
            'Name of product': product,
            'lifecycle_stage': stage,
            'stage_emoji': emoji,
            'age_months': age_months,
            'growth_rate': round(growth_rate * 100, 1),
            'total_volume': total_volume,
            'recent_volume': volume_recent
        })
    
    result_df = pd.DataFrame(product_stages)
    return result_df.sort_values('total_volume', ascending=False).reset_index(drop=True)


@st.cache_data(ttl=3600)
def compute_growth_decomposition(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> dict:
    """
    Decompose volume growth into components: new customers, expansion, churn, price impact, mix impact.
    
    Note: 'Sold' column represents Volume (KG), not monetary revenue.

    Args:
        df_curr: Current period data
        df_prev: Previous period data

    Returns:
        dict: {
            'total_growth': float (KG),
            'total_growth_pct': float,
            'volume_prev': float (KG),
            'volume_curr': float (KG),
            'components': {
                'new_customers': float,
                'expansion': float,
                'churn': float,
                'price_impact': float,
                'mix_impact': float
            },
            'component_pct': {...}
        }
    """
    if df_prev is None or df_prev.empty:
        return None
    
    if df_curr is None or df_curr.empty:
        return None
    
    volume_curr = df_curr['Sold'].sum()  # Volume in KG
    volume_prev = df_prev['Sold'].sum()
    total_growth = volume_curr - volume_prev
    
    # Customer sets
    clients_curr = set(df_curr['Name of client'].unique())
    clients_prev = set(df_prev['Name of client'].unique())
    
    # 1. New Customer Volume
    new_clients = clients_curr - clients_prev
    new_customer_volume = df_curr[df_curr['Name of client'].isin(new_clients)]['Sold'].sum()
    
    # 2. Churned Customer Volume (lost)
    churned_clients = clients_prev - clients_curr
    churn_volume = -df_prev[df_prev['Name of client'].isin(churned_clients)]['Sold'].sum()
    
    # 3. Existing Customer Expansion/Contraction
    retained_clients = clients_curr & clients_prev
    existing_volume_curr = df_curr[df_curr['Name of client'].isin(retained_clients)]['Sold'].sum()
    existing_volume_prev = df_prev[df_prev['Name of client'].isin(retained_clients)]['Sold'].sum()
    expansion_volume = existing_volume_curr - existing_volume_prev
    
    # 4. Price Impact (using average price per unit - less relevant for volume)
    qty_curr = df_curr['Quantity (KG)'].sum()
    qty_prev = df_prev['Quantity (KG)'].sum()
    
    avg_price_curr = volume_curr / qty_curr if qty_curr > 0 else 0
    avg_price_prev = volume_prev / qty_prev if qty_prev > 0 else 0
    
    # Price impact: change in avg volume per order * current quantity
    price_impact = (avg_price_curr - avg_price_prev) * qty_curr if qty_curr > 0 else 0
    
    # 5. Mix Impact (residual - captures product mix changes)
    explained = new_customer_volume + churn_volume + expansion_volume + price_impact
    mix_impact = total_growth - explained
    
    # Calculate percentage contributions
    component_pct = {}
    if total_growth != 0:
        component_pct = {
            'new_customers': round((new_customer_volume / abs(total_growth)) * 100, 1),
            'expansion': round((expansion_volume / abs(total_growth)) * 100, 1),
            'churn': round((churn_volume / abs(total_growth)) * 100, 1),
            'price_impact': round((price_impact / abs(total_growth)) * 100, 1),
            'mix_impact': round((mix_impact / abs(total_growth)) * 100, 1)
        }
    else:
        component_pct = {
            'new_customers': 0,
            'expansion': 0,
            'churn': 0,
            'price_impact': 0,
            'mix_impact': 0
        }
    
    return {
        'total_growth': round(total_growth, 0),
        'total_growth_pct': round((total_growth / volume_prev) * 100, 1) if volume_prev > 0 else 0,
        'volume_prev': round(volume_prev, 0),
        'volume_curr': round(volume_curr, 0),
        'components': {
            'new_customers': round(new_customer_volume, 0),
            'expansion': round(expansion_volume, 0),
            'churn': round(churn_volume, 0),
            'price_impact': round(price_impact, 0),
            'mix_impact': round(mix_impact, 0)
        },
        'component_pct': component_pct
    }


@st.cache_data(ttl=3600)
def compute_launch_velocity(df: pd.DataFrame, min_age_months: int = 3) -> pd.DataFrame:
    """
    Calculate launch velocity for new products (launched in last 12 months).
    
    Note: 'Sold' column represents Volume (KG), not monetary revenue.

    Args:
        df: Full historical data
        min_age_months: Minimum age in months to include (default 3)

    Returns:
        DataFrame columns: ['Name of product', 'launch_date', 'age_months',
                           'velocity_pct', 'velocity_category', 'velocity_emoji',
                           'm1_volume', 'm3_volume', 'current_volume',
                           'm1_customers', 'm3_customers']
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'Name of product', 'launch_date', 'age_months',
            'velocity_pct', 'velocity_category', 'velocity_emoji',
            'm1_volume', 'm3_volume', 'current_volume',
            'm1_customers', 'm3_customers'
        ])
    
    today = df[DEFAULT_DATE_COL].max()
    twelve_months_ago = today - pd.Timedelta(days=365)
    
    # Pre-compute launch dates for all products
    product_launch_dates = df.groupby('Name of product')[DEFAULT_DATE_COL].min()
    
    product_launches = []
    
    for product in df['Name of product'].unique():
        launch_date = product_launch_dates.get(product, today)
        
        # Only products launched in last 12 months
        if launch_date < twelve_months_ago:
            continue
        
        age_months = max(1, ((today - launch_date).days // 30))
        
        # Need at least min_age_months data
        if age_months < min_age_months:
            continue
        
        prod_df = df[df['Name of product'] == product]
        
        # M1, M3 volume windows
        m1_end = launch_date + pd.Timedelta(days=30)
        m3_end = launch_date + pd.Timedelta(days=90)
        
        m1_df = prod_df[(prod_df[DEFAULT_DATE_COL] >= launch_date) & (prod_df[DEFAULT_DATE_COL] < m1_end)]
        m1_volume = m1_df['Sold'].sum()  # Volume in KG
        m1_customers = m1_df['Name of client'].nunique()
        
        m3_df = prod_df[(prod_df[DEFAULT_DATE_COL] >= launch_date) & (prod_df[DEFAULT_DATE_COL] < m3_end)]
        m3_volume = m3_df['Sold'].sum()
        m3_customers = m3_df['Name of client'].nunique()
        
        current_volume = prod_df['Sold'].sum()
        
        # Velocity calculation: (M3 volume - M1 volume) / M1 volume * 100
        if m1_volume > 0:
            velocity = ((m3_volume - m1_volume) / m1_volume) * 100
        else:
            velocity = 0 if m3_volume == 0 else 999  # Strong indicator if volume from nothing
        
        # Categorize velocity
        if velocity >= 100:
            category, emoji = 'Fast', '🚀'
        elif velocity >= 50:
            category, emoji = 'Moderate', '🏃'
        elif velocity >= 0:
            category, emoji = 'Slow', '🐌'
        else:
            category, emoji = 'Declining', '📉'
        
        product_launches.append({
            'Name of product': product,
            'launch_date': launch_date,
            'age_months': age_months,
            'velocity_pct': round(velocity, 1),
            'velocity_category': category,
            'velocity_emoji': emoji,
            'm1_volume': round(m1_volume, 0),
            'm3_volume': round(m3_volume, 0),
            'current_volume': round(current_volume, 0),
            'm1_customers': m1_customers,
            'm3_customers': m3_customers
        })
    
    result_df = pd.DataFrame(product_launches)
    if not result_df.empty:
        result_df = result_df.sort_values('velocity_pct', ascending=False).reset_index(drop=True)
    return result_df

