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
def load_data(file_path_or_buffer, csv_fallback=True, nrows=None) -> pd.DataFrame:
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
            df = pd.read_csv(file_path_or_buffer, nrows=nrows)
        else:
            # Default to Excel - Read ALL sheets
            dfs = pd.read_excel(file_path_or_buffer, nrows=nrows, engine='openpyxl', sheet_name=None)
            
            # If multiple sheets, concat
            if isinstance(dfs, dict):
                # Filter out empty sheets or metadata sheets if needed
                # For now, just concat all frames
                all_frames = []
                for sheet_name, sheet_df in dfs.items():
                    # Add sheet name as column for debugging/filtering if useful
                    # sheet_df['Sheet'] = sheet_name 
                    all_frames.append(sheet_df)
                df = pd.concat(all_frames, ignore_index=True)
            else:
                df = dfs
                
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

    # Year/Month to integers
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    # Clean Month column before coercion
    def clean_month_val(x):
        if isinstance(x, str):
            digits = re.findall(r'\d+', x)
            return int(digits[0]) if digits else x
        return x
        
    df['Month'] = df['Month'].apply(clean_month_val)
    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')

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

    # Synthetic date column 1st day of month (useful for time-series grouping)
    def make_date(r):
        try:
            y = int(r['Year'])
            m = r['Month']
            
            # Handle string months like 'R1', 'M01', etc.
            if isinstance(m, str):
                # Extract digits
                digits = re.findall(r'\d+', m)
                if digits:
                    m = int(digits[0])
                else:
                    # Fallback to 1 if no digits found
                    m = 1
            
            # Ensure m is within 1-12 range logic if needed, or just standard date
            # If m is 0 or >12, date() will raise ValueError, so we catch it
            return dt.date(y, m, 1)
        except Exception:
            return pd.NaT

    df[DEFAULT_DATE_COL] = df.apply(make_date, axis=1)
    df[DEFAULT_DATE_COL] = pd.to_datetime(df[DEFAULT_DATE_COL])

    # Drop rows without date
    df = df[~df[DEFAULT_DATE_COL].isna()].reset_index(drop=True)

    return df
