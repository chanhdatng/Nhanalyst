import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils import DEFAULT_DATE_COL

def fig_top_level(kpis: dict) -> go.Figure:
    monthly = kpis['monthly_series']
    fig = px.line(monthly, x=DEFAULT_DATE_COL, y='Sold', title='Monthly Revenue')
    return fig


def fig_top_products(prod_df: pd.DataFrame, top_n=20) -> go.Figure:
    top = prod_df.head(top_n)
    fig = px.bar(top, x='Name of product', y='revenue', title=f'Top {top_n} Products by Revenue')
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    return fig


def fig_region_map(region_df: pd.DataFrame):
    # If region_df contains country names we can use choropleth; user may need to provide ISO codes
    if 'Country' in region_df.columns:
        agg = region_df.groupby('Country').agg(revenue=('revenue', 'sum')).reset_index()
        try:
            fig = px.choropleth(agg, locations='Country', locationmode='country names', color='revenue', title='Revenue by Country')
            return fig
        except Exception:
            # Fallback: bar chart
            fig = px.bar(agg.sort_values('revenue', ascending=False).head(20), x='Country', y='revenue', title='Revenue by Country (Top 20)')
            return fig
    else:
        return go.Figure()
