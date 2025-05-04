import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import scienceplots

from stats import display_stats
from utils import get_example_data

st.set_page_config(page_title="Data Plotting App")

st.title("Data Plotting App")

# Option to load example data
use_example_data = st.checkbox("Use example data")

if use_example_data:
    df = get_example_data()
    if df is not None:
        st.success("Example data loaded successfully!")
    uploaded_file = None
else:
    # File upload
    uploaded_file = st.file_uploader("Choose a data file", type=["csv", "txt", "xlsx", "xls"])

if uploaded_file is not None or (use_example_data and df is not None):
    if uploaded_file is not None:
        # Determine file type and read accordingly
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            # Let user specify delimiter and handle quoting
            delimiter = st.text_input("Specify delimiter (leave empty for auto-detection)", ",")
            delimiter = delimiter if delimiter else None
                
            # Add options for CSV parsing
            header_option = st.selectbox("Header row", ["First row is header", "No header"])
            header = 0 if header_option == "First row is header" else None
            df = pd.read_table(uploaded_file, delimiter=delimiter, header=header, quotechar='"', dtype=float)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Select columns to plot
    st.subheader("Plot Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.selectbox("Plot Type", ["Line", "Scatter", "Bar", "Histogram", "Box", "Heatmap"])
    
    # Get numeric columns for plotting
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) > 0:
        with col2:
            theme_options = ["Default Matplotlib", "Science"]
            theme_option = st.selectbox("Theme", theme_options)
        
        # Different plot types need different column selections
        if plot_type in ["Line", "Scatter", "Bar"]:
            x_col = st.selectbox("X-axis", df.columns.tolist())
            # Filter out the X column from Y options to prevent duplicates
            y_options = [col for col in numeric_columns if col != x_col]
            
            if not y_options and len(numeric_columns) > 0:
                st.warning(f"You selected '{x_col}' for X-axis. Please choose different columns for Y-axis.")
                # Add back the X column as an option but with a warning
                y_options = numeric_columns
                
            y_cols = st.multiselect("Y-axis", y_options, 
                                   default=[y_options[0]] if y_options else [])
                
            # Warning if they try to select same column for X and Y
            if x_col in y_cols:
                st.warning(f"You've selected '{x_col}' for both X and Y axes. This may cause plotting errors.")
                
            # NEW: Add range sliders for X and Y axes percentages
            st.subheader("Data Range Filters")
            
            col1, col2 = st.columns(2)


            # X-axis range slider
            with col1:
                st.write("#### X-axis Range (Percentiles)")
                x_range = st.slider("X-axis Percentile Range", 0.0, 100.0, (0.0, 100.0), 0.1, 
                                format="%.1f%%")
            
            # Y-axis range slider
            with col2:
                st.write("#### Y-axis Range (Percentiles)")
                y_range = st.slider("Y-axis Percentile Range", 0.0, 100.0, (0.0, 100.0), 0.1,
                                format="%.1f%%")
                
        elif plot_type in ["Histogram", "Box"]:
            y_cols = st.multiselect("Select columns", numeric_columns, 
                                   default=[numeric_columns[0]] if numeric_columns else [])
            x_col = None
            
            # NEW: Add range slider for Y axis only (no X-axis for these plots)
            st.subheader("Data Range Filters")
            st.write("#### Value Range (Percentiles)")
            y_range = st.slider("Value Percentile Range", 0.0, 100.0, (0.0, 100.0), 0.1,
                              format="%.1f%%")
            x_range = (0.0, 100.0)  # Default full range for X
                
        elif plot_type == "Heatmap":
            corr_cols = st.multiselect("Select columns for correlation", numeric_columns, 
                                     default=numeric_columns[:min(5, len(numeric_columns))])
            x_col = None
            y_cols = corr_cols
            # No range sliders needed for heatmap
            x_range = (0.0, 100.0)
            y_range = (0.0, 100.0)
        
        # Plot customization options
        st.subheader("Plot Customization")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_width = st.slider("Figure Width", min_value=4, max_value=40, value=10)
            title = st.text_input("Plot Title", "")
            
            # Axis scale options
            x_scale = st.selectbox("X-axis Scale", ["Linear", "Log"])

            grid = st.checkbox("Show Grid", True)
            
        with col2:
            fig_height = st.slider("Figure Height", min_value=4, max_value=40, value=6)
            legend_pos = st.selectbox("Legend Position", ["best", "upper right", "upper left", "lower right", "lower left"])
            
            # Y-axis scale
            y_scale = st.selectbox("Y-axis Scale", ["Linear", "Log"])

            equal_axes = st.checkbox("Equal Axes Scales", False)

        # Custom axis labels
        col1, col2 = st.columns(2)
        with col1:
            x_label = st.text_input("X-axis Label", "" if x_col is None else x_col)
        with col2:
            y_label = st.text_input("Y-axis Label", "Value")
            
        # Generate plot if columns selected
        if (plot_type in ["Histogram", "Box"] and y_cols) or \
           (plot_type in ["Line", "Scatter", "Bar"] and x_col and y_cols) or \
           (plot_type == "Heatmap" and y_cols):
            
            # Apply the selected theme
            if theme_option == "Science":
                plt.style.use(['science', 'no-latex'])
            else:
                plt.style.use('default')
            
            # NEW: Filter the data based on the percentile ranges
            filtered_df = df.copy()
            
            if plot_type in ["Line", "Scatter", "Bar"]:
                # Filter X-axis data based on percentiles
                if x_range != (0.0, 100.0):
                    x_min = np.percentile(filtered_df[x_col], x_range[0])
                    x_max = np.percentile(filtered_df[x_col], x_range[1])
                    filtered_df = filtered_df[(filtered_df[x_col] >= x_min) & (filtered_df[x_col] <= x_max)]
                
                # Filter Y-axis data based on percentiles for each column
                if y_range != (0.0, 100.0):
                    y_mask = pd.Series(True, index=filtered_df.index)
                    for col in y_cols:
                        y_min = np.percentile(df[col], y_range[0])
                        y_max = np.percentile(df[col], y_range[1])
                        y_mask = y_mask & (filtered_df[col] >= y_min) & (filtered_df[col] <= y_max)
                    filtered_df = filtered_df[y_mask]
            
            elif plot_type in ["Histogram", "Box"] and y_range != (0.0, 100.0):
                # Filter data for histogram/box plots
                # Create a mask that includes rows where ANY of the selected columns are within the range
                mask = pd.Series(False, index=filtered_df.index)
                for col in y_cols:
                    y_min = np.percentile(df[col], y_range[0])
                    y_max = np.percentile(df[col], y_range[1])
                    mask = mask | ((filtered_df[col] >= y_min) & (filtered_df[col] <= y_max))
                filtered_df = filtered_df[mask]
            
            # Check if we still have data after filtering
            if len(filtered_df) == 0:
                st.error("No data points remain after applying the filters. Please adjust your percentile ranges.")
                st.stop()
                
            # Display how many points were kept after filtering
            st.info(f"Using {len(filtered_df)} out of {len(df)} data points ({len(filtered_df)/len(df)*100:.1f}%) after filtering.")
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Set axis scales
            if x_scale == "Log" and plot_type != "Heatmap":
                ax.set_xscale('log')
            if y_scale == "Log" and plot_type != "Heatmap":
                ax.set_yscale('log')
            
            # Set equal axes if selected
            if equal_axes and plot_type not in ["Heatmap", "Box", "Histogram"]:
                ax.set_aspect('equal')
            
            try:
                if plot_type == "Line":
                    # Use matplotlib directly instead of pandas plot when x_col is in y_cols
                    if x_col in y_cols:
                        for col in y_cols:
                            if col == x_col:
                                continue  # Skip plotting a column against itself
                            ax.plot(filtered_df[x_col], filtered_df[col], label=col)
                        ax.grid(grid)
                        if title:
                            ax.set_title(title)
                        # Set custom axis labels
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.legend(loc=legend_pos)
                    else:
                        # Normal case: use pandas plot
                        for col in y_cols:
                            filtered_df.plot(kind='line', x=x_col, y=col, ax=ax, grid=grid)
                        # Set custom axis labels
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                elif plot_type == "Scatter":
                    # Use matplotlib directly when x_col is in y_cols
                    if x_col in y_cols:
                        for col in y_cols:
                            if col == x_col:
                                continue  # Skip plotting a column against itself
                            ax.scatter(filtered_df[x_col], filtered_df[col], label=col)
                        ax.grid(grid)
                        if title:
                            ax.set_title(title)
                        # Set custom axis labels
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.legend(loc=legend_pos)
                    else:
                        # Normal case
                        for col in y_cols:
                            filtered_df.plot(kind='scatter', x=x_col, y=col, ax=ax, grid=grid)
                        # Set custom axis labels
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                elif plot_type == "Bar":
                    # Use matplotlib directly when x_col is in y_cols
                    if x_col in y_cols:
                        for col in y_cols:
                            if col == x_col:
                                continue  # Skip plotting a column against itself
                            ax.bar(filtered_df[x_col], filtered_df[col], label=col, alpha=0.7)
                        ax.grid(grid)
                        if title:
                            ax.set_title(title)
                        # Set custom axis labels
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.legend(loc=legend_pos)
                    else:
                        # Normal case
                        for col in y_cols:
                            filtered_df.plot(kind='bar', x=x_col, y=col, ax=ax, grid=grid)
                        # Set custom axis labels
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                elif plot_type == "Histogram":
                    for col in y_cols:
                        ax.hist(filtered_df[col].dropna(), alpha=0.7, label=col)
                    ax.grid(grid)
                    # Set custom axis labels
                    ax.set_xlabel("Values")
                    ax.set_ylabel("Frequency")
                    ax.legend(loc=legend_pos)
                elif plot_type == "Box":
                    ax.boxplot([filtered_df[col].dropna() for col in y_cols], labels=y_cols)
                    ax.grid(grid)
                    # Set custom axis labels
                    ax.set_xlabel("Columns")
                    ax.set_ylabel("Values")
                elif plot_type == "Heatmap":
                    corr = filtered_df[y_cols].corr()
                    im = ax.imshow(corr, cmap='coolwarm')
                    plt.colorbar(im, ax=ax)
                    # Add correlation values in the cells
                    for i in range(len(corr.columns)):
                        for j in range(len(corr.columns)):
                            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                    ax.set_yticklabels(corr.columns)
                    # Set custom axis labels
                    ax.set_xlabel("Columns")
                    ax.set_ylabel("Columns")
                
                if title:
                    ax.set_title(title)

                st.pyplot(fig)
                
                # Show statistics only if checkbox is checked
                show_stats = st.checkbox("Show Statistics", False)
                
                if show_stats:
                    display_stats(plot_type, filtered_df, x_col, y_cols)

                
                # Export options
                st.subheader("Export Plot")
                
                col1, col2, col3 = st.columns(3)
                
                # Function to create download link for different formats
                def get_download_link(fig, format_type):
                    buf = io.BytesIO()
                    fig.savefig(buf, format=format_type, bbox_inches='tight')
                    buf.seek(0)
                    return base64.b64encode(buf.read()).decode()
                
                with col1:
                    st.download_button(label="Export as PNG", data=get_download_link(fig, 'png'), file_name="plot.png")
                with col2:
                    st.download_button(label="Export as SVG", data=get_download_link(fig, 'svg'), file_name="plot.svg")
                with col3:
                    st.download_button(label="Export as PDF", data=get_download_link(fig, 'pdf'), file_name="plot.pdf")
                
            except Exception as e:
                st.error(f"Error generating plot: {e}")
                st.exception(e)
        else:
            if plot_type in ["Line", "Scatter", "Bar"]:
                st.info("Please select both X and Y columns to generate the plot.")
            else:
                st.info("Please select at least one column to generate the plot.")
    else:
        st.error("No numeric columns found in the dataset for plotting.")
else:
    st.info("Please upload a data file or use the example data to get started.")

# Add a footer with date information
st.markdown("""
---
*Data Plotting App | mxdbck | Last Updated: 2025-05-03 16:24:33*
""")