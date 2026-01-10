import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import streamlit as st
import re
from src.utils import DEFAULT_DATE_COL

# Expected columns based on your schema
EXPECTED_COLS = [
    'Year', 'Month', 'Name of client', 'Channel by Sales Person',
    'Region', 'Country', 'Name of product', 'Kind of fruit', 'SKU',
    'Type of product', 'Sold', 'Quantity (KG)'
]

@st.cache_data(show_spinner=False)
def load_data(file_path_or_buffer, csv_fallback=True, nrows=None, read_all_sheets=False, usecols=None) -> pd.DataFrame:
    """Load Excel or CSV robustly.
    - file_path_or_buffer: path to .xlsx/.csv OR a file-like object (buffer)
    - nrows: optional for sampling
    Returns full dataframe.
    """
    # Helper to detect extension if it's a file path
    is_path = isinstance(file_path_or_buffer, (str, Path))
    
    if is_path:
        p = Path(file_path_or_buffer)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path_or_buffer}")
        suffix = p.suffix.lower()
    else:
        # It's a buffer (e.g. from streamlit uploader). 
        # We try to guess from the .name attribute if available, else try read_excel then read_csv
        suffix = Path(file_path_or_buffer.name).suffix.lower() if hasattr(file_path_or_buffer, 'name') else ''

    try:
        if suffix == '.csv':
            df = pd.read_csv(file_path_or_buffer, nrows=nrows, usecols=usecols)
        else:
            # Default to Excel - read only first sheet unless read_all_sheets=True
            if read_all_sheets:
                dfs = pd.read_excel(file_path_or_buffer, nrows=nrows, engine='openpyxl', sheet_name=None, usecols=usecols)
                if isinstance(dfs, dict):
                    all_frames = []
                    for sheet_name, sheet_df in dfs.items():
                        all_frames.append(sheet_df)
                    df = pd.concat(all_frames, ignore_index=True)
                else:
                    df = dfs
            else:
                # Read only the first sheet which is far faster for large Excel files
                df = pd.read_excel(file_path_or_buffer, nrows=nrows, engine='openpyxl', sheet_name=0, usecols=usecols)
                
    except Exception as e:
        # Fallback logic could go here, for now just re-raise
        raise ValueError(f"Could not load data. Error: {e}")

    # Standardize column names by stripping
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean & canonicalize the incoming dataframe. Returns cleaned df.
    Steps:
      - Ensure expected columns exist (attempt fuzzy match)
      - Cast types
      - Create a synthetic date column: Year/Month -> 1st of month
      - Fill NAs with sensible defaults
      - Standardize text columns
    """
    df = df.copy()

    # --- fuzzy column mapping if user has slightly different headers
    # Priority mapping for known variations
    known_mappings = {
        'Sold quantity (KG)': 'Quantity (KG)',
        'Sales': 'Sold',
        'Revenue': 'Sold',
        'Amount': 'Sold',
        'Channel': 'Channel by Sales Person'
    }
    df = df.rename(columns=known_mappings)

    col_map = {}
    cols = list(df.columns)
    for expected in EXPECTED_COLS:
        for c in cols:
            # Exact match already handled by rename or existence
            if c in df.columns:
                continue
            # Fuzzy match
            if c.lower().replace(' ', '') == expected.lower().replace(' ', ''):
                col_map[c] = expected
                break
    if col_map:
        df = df.rename(columns=col_map)

    # Ensure all expected exist; if not, create placeholders
    for c in EXPECTED_COLS:
        if c not in df.columns:
            # If 'Sold' is missing, likely no revenue column. Warn or just 0.
            # print(f"Warning: Column '{c}' not found. Filling with defaults.") 
            df[c] = pd.NA

    # Cast numeric columns
    df['Quantity (KG)'] = pd.to_numeric(df['Quantity (KG)'], errors='coerce').fillna(0.0)
    # Use Quantity as Sold (Revenue) as per user request
    df['Sold'] = df['Quantity (KG)']

    # Year/Month to integers (vectorized where possible)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Vectorized month extraction: extract digits from strings and coerce to numeric
    # This avoids per-row Python loops which are slow on large frames
    if 'Month' in df.columns:
        # Convert to string first so str.extract works for mixed types
        month_str = df['Month'].astype(str)
        extracted = month_str.str.extract(r'(\d+)', expand=False)
        df['Month'] = pd.to_numeric(extracted, errors='coerce')
    else:
        df['Month'] = pd.NA

    # Fill missing client names
    df['Name of client'] = df['Name of client'].fillna('Unknown client').astype(str)

    # Standardize text columns: strip, title-case where appropriate
    text_cols = ['Channel by Sales Person', 'Region', 'Country', 'Kind of fruit', 'SKU', 'Type of product']
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str).str.strip()
            # Normalize Channel to Title Case to merge "Retail" vs "retail"
            if c == 'Channel by Sales Person':
                df[c] = df[c].str.title()
            
    if 'Name of product' in df.columns:
        # User request: remove 'ANDROS PROFESSIONAL' globally
        df['Name of product'] = df['Name of product'].fillna('Unknown').astype(str).str.replace('ANDROS PROFESSIONAL', '', case=False).str.strip()
        # Clean extra spaces
        df['Name of product'] = df['Name of product'].str.replace(r'\s+', ' ', regex=True)

    # Synthetic date column 1st day of month (vectorized)
    # Fill missing months with 1 (January) as a sensible default for grouping
    month_filled = df['Month'].fillna(1).astype(int)
    # Use pandas vectorized datetime constructor from dict to avoid apply
    df[DEFAULT_DATE_COL] = pd.to_datetime(
        {
            'year': df['Year'].astype('Int64'),
            'month': month_filled,
            'day': 1
        },
        errors='coerce'
    )

    # Drop rows without date (invalid/missing Year or Month)
    df = df[~df[DEFAULT_DATE_COL].isna()].reset_index(drop=True)

    return df
