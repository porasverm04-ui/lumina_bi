import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("📊 Data Distribution Dashboard")

# --- CHECK DATA ---
if "cleaned_df" in st.session_state:
    df = st.session_state["cleaned_df"]

    # Create the 50/50 screen split
    col1, col2 = st.columns(2)

    # LEFT COLUMN: NUMERIC DATA (HISTOGRAMS & CORRELATIONS)
    with col1:
        st.subheader("📈 Numeric Analysis")
        numeric_df = df.select_dtypes(include=['number'])
        num_cols = numeric_df.columns.tolist()

        if len(num_cols) > 0:
            # 1. Select X-Axis Column
            x_col = st.selectbox("Select X-Axis (Numeric)", num_cols, key="num_x_select")

            # 2. Check for strong correlations (<= -0.5 or >= 0.5) with other columns
            corr_series = numeric_df.corr()[x_col].drop(labels=[x_col], errors='ignore')
            high_corr = corr_series[(corr_series >= 0.5) | (corr_series <= -0.5)]

            # Display an alert if strong correlations exist for the chosen X column
            if not high_corr.empty:
                corr_list = [f"**{col}** ({val:.2f})" for col, val in high_corr.items()]
                st.info(f"⚡ Highly correlated with `{x_col}`: " + ", ".join(corr_list))

            # 3. Choose Analysis Mode (if at least 2 numeric columns exist)
            if len(num_cols) > 1:
                mode = st.radio(
                    "Select Y-Axis Mode:",
                    ["Count (Histogram)", "Compare with another column (Correlation)"],
                    horizontal=True,
                    key="num_mode"
                )
            else:
                mode = "Count (Histogram)"

            # 4. Render the appropriate chart
            if mode == "Count (Histogram)":
                fig, ax = plt.subplots()
                sns.histplot(df[x_col], kde=True, ax=ax, color='royalblue')
                ax.set_title(f"Distribution of {x_col}")
                st.pyplot(fig)

            else:
                # Filter out x_col so you don't correlate a column with itself
                y_options = [c for c in num_cols if c != x_col]

                # Add a 🔥 badge in the dropdown next to highly correlated columns
                def format_y_label(col_name):
                    if col_name in high_corr:
                        return f"🔥 {col_name} (r = {high_corr[col_name]:.2f})"
                    return f"{col_name} (r = {corr_series.get(col_name, 0):.2f})"

                y_col = st.selectbox(
                    "Select Y-Axis (Numeric)", 
                    y_options, 
                    format_func=format_y_label,
                    key="num_y_select"
                )

                r_val = corr_series.get(y_col, 0)
                if r_val >= 0.5 or r_val <= -0.5:
                    st.success(f"🔥 **High Correlation Detected!** (r = {r_val:.2f})")
                else:
                    st.caption(f"Weak/Moderate Correlation (r = {r_val:.2f})")

                # Plot Seaborn Scatter + Regression Line
                fig, ax = plt.subplots()
                sns.regplot(
                    data=df, x=x_col, y=y_col, ax=ax,
                    color='royalblue',
                    line_kws={"color": "crimson"}
                )
                ax.set_title(f"{x_col} vs {y_col} (r = {r_val:.2f})")
                st.pyplot(fig)

        else:
            st.warning("No numeric columns available for analysis")

    # RIGHT COLUMN: CATEGORICAL DATA
    with col2:
        st.subheader("🥧 Category Analysis")
        
        # Filter out high-cardinality garbage (like IDs or full text)
        cat_columns = [col for col in df.columns if df[col].nunique() <= 100]

        if len(cat_columns) > 0:
            cat_col = st.selectbox("Select Categorical Column", cat_columns, key="cat_select")
            unique_count = df[cat_col].nunique()
            
            # Prepare the math for Plotly
            chart_data = df[cat_col].value_counts().reset_index()
            chart_data.columns = [cat_col, 'Count']

            # THE BRAIN: Pie Chart vs Bar Chart
            if 0 < unique_count <= 30:
                fig2 = px.pie(chart_data, names=cat_col, values='Count', hole=0.3)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig2 = px.bar(chart_data, x=cat_col, y='Count', color=cat_col)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No categorical columns available.")

else:
    st.warning("⚠️ Please upload and clean data first on the Clean Data page.")
    st.stop()