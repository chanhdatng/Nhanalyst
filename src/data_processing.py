import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import streamlit as st
import re
from src.utils import DEFAULT_DATE_COL

# Expected columns based on your schema
EXPECTED_COLS = [
    'Year', 'Month', 'Name of client', 'New Channel',
    'Region', 'Country', 'Name of product', 'Kind of fruit', 'SKU',
    'Type of product', 'Sold', 'Quantity (KG)'
]

# Column name variations mapping (variation -> canonical name)
COLUMN_ALIASES = {
    # Month variations
    'Month1': 'Month',
    'Month2': 'Month',
    # Channel variations - only map case variations of 'New Channel'
    # Do NOT map 'Channel' or 'Channel by Sales Person' to avoid conflicts
    'new channel': 'New Channel',
    'NEW CHANNEL': 'New Channel',
    'NewChannel': 'New Channel',
    'newchannel': 'New Channel',
    # Quantity variations
    'Sold quantity (KG)-B2026': 'Quantity (KG)',
    'Sold quantity (KG)': 'Quantity (KG)',
    'ShippedQtyByNetWeightKG-B2026': 'Quantity (KG)',
    'Quantity': 'Quantity (KG)',
    'Qty (KG)': 'Quantity (KG)',
    # Sales/Revenue variations
    'Sales': 'Sold',
    'Revenue': 'Sold',
    'Amount': 'Sold',
    'Gross Sales 3rd Party-B2026': 'Sold',
    'Net customer sales-B2026': 'Sold',
    # Client variations
    'Client': 'Name of client',
    'Customer': 'Name of client',
    'Customer Name': 'Name of client',
}

# Key columns to detect header row
HEADER_MARKERS = ['Year', 'Month', 'Month1', 'Name of client', 'Client', 'Region', 'Country', 'SKU', 'New Channel']

def _detect_header_row(file_path_or_buffer, suffix: str, max_rows: int = 10) -> int:
    """Scan first few rows to find the actual header row containing known column names."""
    try:
        if suffix == '.csv':
            preview = pd.read_csv(file_path_or_buffer, nrows=max_rows, header=None)
        else:
            preview = pd.read_excel(file_path_or_buffer, nrows=max_rows, header=None, engine='openpyxl', sheet_name=0)

        # Reset buffer position if it's a file-like object
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)

        # Check each row for header markers
        for idx in range(len(preview)):
            row_values = [str(v).strip() for v in preview.iloc[idx].values]
            matches = sum(1 for marker in HEADER_MARKERS if marker in row_values)
            if matches >= 3:  # At least 3 known columns found
                return idx
    except Exception:
        pass

    return 0  # Default to first row


@st.cache_data(show_spinner=False)
def load_data(file_path_or_buffer, csv_fallback=True, nrows=None, read_all_sheets=False, usecols=None) -> pd.DataFrame:
    """Load Excel or CSV robustly with auto-detection of header row.
    - file_path_or_buffer: path to .xlsx/.csv OR a file-like object (buffer)
    - nrows: optional for sampling
    Returns full dataframe with standardized column names.
    """
    is_path = isinstance(file_path_or_buffer, (str, Path))

    if is_path:
        p = Path(file_path_or_buffer)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path_or_buffer}")
        suffix = p.suffix.lower()
    else:
        suffix = Path(file_path_or_buffer.name).suffix.lower() if hasattr(file_path_or_buffer, 'name') else ''

    # Auto-detect header row
    header_row = _detect_header_row(file_path_or_buffer, suffix)

    try:
        if suffix == '.csv':
            df = pd.read_csv(file_path_or_buffer, nrows=nrows, usecols=usecols, header=header_row)
        else:
            if read_all_sheets:
                dfs = pd.read_excel(file_path_or_buffer, nrows=nrows, engine='openpyxl',
                                   sheet_name=None, usecols=usecols, header=header_row)
                if isinstance(dfs, dict):
                    all_frames = list(dfs.values())
                    df = pd.concat(all_frames, ignore_index=True)
                else:
                    df = dfs
            else:
                df = pd.read_excel(file_path_or_buffer, nrows=nrows, engine='openpyxl',
                                  sheet_name=0, usecols=usecols, header=header_row)

    except Exception as e:
        raise ValueError(f"Could not load data. Error: {e}")

    # Standardize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Handle duplicate columns BEFORE applying aliases: keep first occurrence, rename others
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols

    # Apply column aliases to canonical names
    df = df.rename(columns=COLUMN_ALIASES)

    # Handle any NEW duplicates created by alias mapping (e.g., Month1->Month, Month2->Month)
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols

    return df


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean & canonicalize the incoming dataframe. Returns cleaned df.
    Steps:
      - Ensure expected columns exist (apply aliases already done in load_data)
      - Cast types
      - Create a synthetic date column: Year/Month -> 1st of month
      - Fill NAs with sensible defaults
      - Standardize text columns
    """
    df = df.copy()

    # Apply aliases again in case df came from elsewhere (not load_data)
    df = df.rename(columns=COLUMN_ALIASES)

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
        # Use .iloc[:, 0] if DataFrame returned (due to duplicate columns)
        month_col = df['Month']
        if isinstance(month_col, pd.DataFrame):
            month_col = month_col.iloc[:, 0]
        month_str = month_col.astype(str)
        extracted = month_str.str.extract(r'(\d+)', expand=False)
        df['Month'] = pd.to_numeric(extracted, errors='coerce')
    else:
        df['Month'] = pd.NA

    # Fill missing client names
    df['Name of client'] = df['Name of client'].fillna('Unknown client').astype(str)

    # Standardize text columns: strip, title-case where appropriate
    text_cols = ['New Channel', 'Region', 'Country', 'Kind of fruit', 'SKU', 'Type of product']
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str).str.strip()
            # Normalize Channel to Title Case to merge "Retail" vs "retail"
            if c == 'New Channel':
                df[c] = df[c].str.title()
            
    if 'Name of product' in df.columns:
        # User request: remove 'ANDROS PROFESSIONAL' globally
        df['Name of product'] = df['Name of product'].fillna('Unknown').astype(str).str.replace('ANDROS PROFESSIONAL', '', case=False).str.strip()
        # Clean extra spaces
        df['Name of product'] = df['Name of product'].str.replace(r'\s+', ' ', regex=True)

    # Synthetic date column 1st day of month (vectorized)
    # Fill missing months with 1 (January) as a sensible default for grouping
    # Use Int64 (nullable integer) to handle any remaining NA values gracefully
    month_filled = pd.to_numeric(df['Month'], errors='coerce').fillna(1).astype('Int64')
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
