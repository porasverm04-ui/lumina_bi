import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
if "cleaned_df" in st.session_state:
    st.subheader("📈 Correlation Analysis")
    df = st.session_state["cleaned_df"]

    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.shape[1] > 1:

        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        positive_pairs = []
        negative_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                corr_value = corr_matrix.iloc[i, j]
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]

                if corr_value >= 0.5:
                    positive_pairs.append((col1, col2, corr_value))

                elif corr_value <= -0.5:
                    negative_pairs.append((col1, col2, corr_value))
        # --- SORT & TAKE TOP 5 ---
        positive_pairs = sorted(positive_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
        negative_pairs = sorted(negative_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
        # --- POSITIVE ---
        if positive_pairs:
            st.subheader("📈 Positive Correlation (≥ 0.5)")
            for col1, col2, corr_value in positive_pairs:
                st.write(f"**{col1} vs {col2}** → {corr_value:.2f}")

                fig, ax = plt.subplots()
                sns.regplot(x=df[col1], y=df[col2], ax=ax)
                ax.set_title(f"{col1} vs {col2} (Positive)")
                st.pyplot(fig)

        # --- NEGATIVE ---
        if negative_pairs:
            st.subheader("📉 Negative Correlation (≤ -0.5)")
            for col1, col2, corr_value in negative_pairs:
                st.write(f"**{col1} vs {col2}** → {corr_value:.2f}")

                fig, ax = plt.subplots()
                sns.regplot(x=df[col1], y=df[col2], ax=ax)
                ax.set_title(f"{col1} vs {col2} (Negative)")
                st.pyplot(fig)

        if not positive_pairs and not negative_pairs:
            st.info("No strong correlations found (|corr| ≥ 0.5)")

    else:
        st.warning("Not enough numeric columns")



else:
    st.warning("⚠️ Please upload and clean data first from Clean Data page")
    st.stop()



    