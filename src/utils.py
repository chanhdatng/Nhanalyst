import os
import json
import pandas as pd
import numpy as np

# Optional: OpenAI for AI insights
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

DEFAULT_DATE_COL = 'date__ym'  # synthetic date column

def filter_by_date(df: pd.DataFrame, years: list, months: list) -> pd.DataFrame:
    """Filter DF by list of years and optional list of months."""
    mask = df['Year'].isin(years)
    if months:
        # Check if 'Month' column is numeric or string match
        # If months are selected as names (Jan, Feb...), map them or use numbers if input is numbers
        mask &= df['Month'].isin(months)
    return df.loc[mask]

def calculate_growth(current_val, prev_val):
    if prev_val == 0 or pd.isna(prev_val):
        return None
    return (current_val - prev_val) / prev_val

def ai_insights_summary(kpis: dict, top_clients_df: pd.DataFrame, top_products_df: pd.DataFrame, openai_api_key: str = None) -> str:
    """Generate short insights. If OpenAI key provided and openai package installed, will call the API.
    Otherwise returns a template summary you can use.
    """
    summary = []
    summary.append(f"Total revenue: {kpis.get('total_revenue'):,}")
    summary.append(f"Total KG sold: {kpis.get('total_kg'):,}")
    if kpis.get('yoy_growth') is not None:
        summary.append(f"YoY growth: {kpis.get('yoy_growth'):.2%}")
    if kpis.get('mom_growth') is not None:
        summary.append(f"MoM growth: {kpis.get('mom_growth'):.2%}")

    # top product
    if kpis.get('top_product'):
        tp = list(kpis['top_product'].items())[0]
        summary.append(f"Top product: {tp[0]} ({tp[1]:,.2f} revenue)")

    # basic templated insights
    templated = '\n'.join(summary)

    if OPENAI_AVAILABLE and openai_api_key:
        openai.api_key = openai_api_key
        prompt = f"You are an expert data analyst. Summarize the following KPI facts and produce 5 action-oriented insights and 3 risks:\n{templated}\nTop 5 clients by revenue:\n{top_clients_df.head(5).to_dict(orient='records')}\nTop 5 products:\n{top_products_df.head(5).to_dict(orient='records')}"
        try:
            resp = openai.Completion.create(
                model='gpt-4o-mini',
                prompt=prompt,
                max_tokens=400,
                temperature=0.2
            )
            return resp.choices[0].text.strip()
        except Exception as e:
            return templated + "\n\nNOTE: OpenAI call failed: " + str(e)
    else:
        # Return a readable insight template
        insight_text = templated + "\n\nSuggested insights (template):\n1. Focus on top 10 clients who contribute majority revenue; run promotions to retain them.\n2. Investigate underperforming regions and reallocate sales effort.\n3. Check seasonality for top fruit types and align inventory.\n4. Evaluate Channel performance and prioritize high AOV channels.\n5. Build RFM segments for targeted campaigns."
        return insight_text

def export_reports(out_dir: str, kpis: dict, client_df: pd.DataFrame, prod_df: pd.DataFrame, region_df: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    # Save main KPIs as JSON
    with open(os.path.join(out_dir, 'kpis.json'), 'w') as f:
        json.dump({k:v if not hasattr(v, 'to_dict') else None for k,v in kpis.items()}, f, default=str)

    client_df.to_csv(os.path.join(out_dir, 'clients.csv'), index=False)
    prod_df.to_csv(os.path.join(out_dir, 'products.csv'), index=False)
    region_df.to_csv(os.path.join(out_dir, 'regions.csv'), index=False)
    if 'monthly_series' in kpis:
        kpis['monthly_series'].to_csv(os.path.join(out_dir, 'monthly_series.csv'), index=False)

    print(f"Reports exported to {out_dir}")
