import streamlit as st
import pandas as pd

def display_stats(plot_type, filtered_df, x_col, y_cols):
    st.subheader("Data Statistics")

    # Create tabs for different statistics
    stats_tab1, stats_tab2 = st.tabs(["Basic Stats", "Correlations"])

    with stats_tab1:
        if plot_type in ["Line", "Scatter", "Bar"]:
            # For X-Y plots, show stats for both X and Y columns
            st.write("#### X-axis Statistics")
            st.write(filtered_df[x_col].describe().to_frame().T)
            
            st.write("#### Y-axis Statistics")
            y_stats = filtered_df[y_cols].describe().T
            st.write(y_stats)
            
            # Add range (max-min) to the statistics
            st.write("#### Value Ranges")
            ranges = pd.DataFrame({
                'Column': [x_col] + y_cols,
                'Range (Max-Min)': [filtered_df[x_col].max() - filtered_df[x_col].min()] + 
                                [filtered_df[col].max() - filtered_df[col].min() for col in y_cols],
                'IQR (Q3-Q1)': [filtered_df[x_col].quantile(0.75) - filtered_df[x_col].quantile(0.25)] + 
                            [filtered_df[col].quantile(0.75) - filtered_df[col].quantile(0.25) for col in y_cols]
            }).set_index('Column')
            st.write(ranges)
            
        elif plot_type in ["Histogram", "Box"]:
            # For single variable plots
            st.write(filtered_df[y_cols].describe().T)
            
            # Add range and IQR
            st.write("#### Value Ranges")
            ranges = pd.DataFrame({
                'Column': y_cols,
                'Range (Max-Min)': [filtered_df[col].max() - filtered_df[col].min() for col in y_cols],
                'IQR (Q3-Q1)': [filtered_df[col].quantile(0.75) - filtered_df[col].quantile(0.25) for col in y_cols]
            }).set_index('Column')
            st.write(ranges)
            
        elif plot_type == "Heatmap":
            st.write(filtered_df[y_cols].describe().T)

    with stats_tab2:
        # Correlation information
        if plot_type in ["Line", "Scatter", "Bar"]:
            columns_to_correlate = [x_col] + y_cols
        else:
            columns_to_correlate = y_cols
        
        if len(columns_to_correlate) > 1:
            corr_matrix = filtered_df[columns_to_correlate].corr()
            st.write("#### Pearson Correlation Matrix")
            st.write(corr_matrix)
            
            if len(columns_to_correlate) <= 10:  # Spearman can be computationally intensive
                try:
                    spearman_corr = filtered_df[columns_to_correlate].corr(method='spearman')
                    st.write("#### Spearman Rank Correlation")
                    st.write(spearman_corr)
                except:
                    st.warning("Could not compute Spearman correlation.")
        else:
            st.info("Select more than one column to view correlations.")