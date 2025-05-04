import os
import numpy as np
import pandas as pd
import streamlit as st

# Function to get default example data
def get_example_data():
    try:
        # Check if example.csv exists in the same directory
        if os.path.exists("example.csv"):
            return pd.read_csv("example.csv")
        else:
            # Create some example data if file doesn't exist
            x = np.linspace(0, 50, 1000)
            df = pd.DataFrame({
                'x': x,
                'sin(x)': np.sin(x),
                'cos(x)': np.cos(x),
                'exp(x/10)': np.exp(x/10)
            })
            return df
    except Exception as e:
        st.error(f"Error loading example data: {e}")
        return None
    