import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


# Streamlit configuration
st.set_page_config(page_title="AI Email Marketing Platform", page_icon="📧", layout="wide")
st.title("📬 AI-Driven Email Marketing Platform")
st.markdown("Upload customer data to analyze engagement, segment users, and generate smart email content.")

# File uploader
file = st.file_uploader("Upload CSV file with customer data", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("✅ File uploaded successfully!")
    st.write("### 🔍 Data Preview:")
    st.dataframe(df.head())

    # Check if required columns are present
    if 'Email Opened' in df.columns and 'Discount Offered' in df.columns:
        # Preprocessing
        features = ['Discount Offered']
        X = df[features]
        y = df['Email Opened']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predict engagement probabilities
        df['Engagement Probability'] = model.predict_proba(X)[:, 1]
        st.subheader("📈 Predicted Engagement Probabilities")
        st.dataframe(df[['Customer ID', 'Discount Offered', 'Email Opened', 'Engagement Probability']].head())

        # Clustering users based on engagement
        st.subheader("👥 Customer Segmentation (KMeans)")
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Segment'] = kmeans.fit_predict(df[['Engagement Probability']])
        st.write("### 📊 Segment Distribution")
        st.write(df['Segment'].value_counts().rename("Users").reset_index(names=["Segment"]))

        # Download segmented data
        st.download_button(
            label="⬇️ Download Segmented Data",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="segmented_customers.csv",
            mime="text/csv"
        )

        # Generate dummy personalized messages
        st.subheader("✉️ Sample Personalized Emails")
        for i in range(3):
            segment_df = df[df['Segment'] == i]
            sample_msg = (
                f"Hi! As a valued customer, we’ve curated a special offer for you "
                f"based on your past interest. Check it out now!"
            )
            st.markdown(f"**Segment {i} ({len(segment_df)} users)**: {sample_msg}")
    else:
        st.warning("⚠️ Required columns 'Email Opened' and 'Discount Offered' not found in CSV.")

else:
    st.info("📂 Please upload a CSV file to begin.")
