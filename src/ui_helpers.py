import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #2E3192 0%, #1BFFFF 100%); /* Dark Blue to Cyan Gradient */
        border: none;
        padding: 15px 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #f0f0f0; /* Light text for dark bg */
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px;
        color: #e0e0e0; /* Lighter delta */
        background-color: rgba(255,255,255,0.2);
        padding: 2px 8px;
        border-radius: 4px;
    }
    
    /* Tabs - dark mode compatible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        font-weight: 600;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F0F4FF;
        color: #1E90FF !important;
    }
    /* Ensure selected tab text is visible */
    .stTabs [aria-selected="true"] p {
        color: #1E90FF !important;
    }
    </style>
    """, unsafe_allow_html=True)

def checkbox_filter(label, options, key_prefix, default_selected=None, expanded=False):
    """Custom multi-select using checkboxes inside an expander."""
    selected = []
    with st.sidebar.expander(label, expanded=expanded):
        # Select All toggle could be added here, but sticking to simple list for now
        
        for val in options:
            # Create a unique key for each checkbox
            k = f"{key_prefix}_{val}"
            # Determine default value
            is_checked = True
            if default_selected is not None:
                    is_checked = (val in default_selected)
            
            if st.checkbox(str(val), value=is_checked, key=k):
                selected.append(val)
    return selected
