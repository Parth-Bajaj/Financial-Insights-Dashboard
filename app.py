# app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# --- Optional: Install required packages if needed ---
# !pip install streamlit transformers torch seaborn matplotlib

# Create a sample CSV with financial data (for demo purposes)
csv_data = {
    "Company": ["Tata Steel", "Infosys", "Reliance", "Tata Steel", "Infosys", "Reliance"],
    "Metric": ["Revenue", "Revenue", "Revenue", "Net Profit", "Net Profit", "Net Profit"],
    "Value": [24335.30, 14862.50, 98702.70, 3923.00, 24108.00, 79020.00]
}
df = pd.DataFrame(csv_data)
df.to_csv("cleaned_financial_data.csv", index=False)
df = pd.read_csv("cleaned_financial_data.csv")

# --- Streamlit UI Setup ---
st.title("Financial Insights Dashboard")
st.markdown("Explore financial trends and ask questions about company performance.")

# Sidebar: Company and Metric selection
st.sidebar.header("Select Options")
company = st.sidebar.selectbox("Select Company", df["Company"].unique())
metric = st.sidebar.selectbox("Select Metric", df["Metric"].unique())

# Filter data based on selection and display
filtered_df = df[(df["Company"] == company) & (df["Metric"] == metric)]
st.write(f"### {company} - {metric}")
if not filtered_df.empty:
    value = filtered_df["Value"].iloc[0]
    st.write(f"**Value:** {value} Billion ₹")
else:
    st.write("No data available for the selected criteria.")

# Visualization: Comparison chart for the selected metric across companies
st.write("### Comparison Chart")
df_metric = df[df["Metric"] == metric]
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Company", y="Value", data=df_metric, ax=ax)
ax.set_title(f"{metric} Comparison Across Companies")
ax.set_ylabel("Value (in Billion ₹)")
ax.set_xlabel("Company")
st.pyplot(fig)

# --- Load Fine-Tuned FinBERT Model ---
model_path = "finbert_finetuned_model"

# Check that the model folder exists
if not os.path.exists(model_path):
    st.error(f"Model folder '{model_path}' not found. Please run the finetuning script first.")
    st.stop()

# Load tokenizer and model using local files only
try:
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Q&A Simulation ---
st.write("### Ask a Financial Question")
user_query = st.text_input("Enter your question:")

if st.button("Get Insight"):
    if not user_query:
        st.write("Please enter a question.")
    else:
        # Tokenize the query and get model predictions
        inputs = tokenizer(user_query, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().numpy()
        predicted_label = logits.argmax(axis=-1)[0]
        
        # For this demo, simulate a response based on keywords in the query
        if "revenue" in user_query.lower():
            response = f"{company}'s revenue is {value} Billion ₹, indicating its market scale."
        elif "net profit" in user_query.lower():
            response = f"{company}'s net profit is {value} Billion ₹, reflecting its profitability."
        else:
            response = "The financial trend suggests steady performance. (Simulated response)"
        st.write(response)
