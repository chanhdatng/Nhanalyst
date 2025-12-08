import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from src.utils import DEFAULT_DATE_COL

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
