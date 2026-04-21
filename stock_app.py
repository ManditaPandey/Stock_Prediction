import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from chatbot_app import chatbot_ui
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        

load_css("style.css")

import os
print(os.getcwd())
# -----------------------
# 1️⃣ Create Dummy Stock Data if not exists
# -----------------------
def create_dummy_stock_data():
    stocks=["AAPL", "TSLA","GOOG","MSFT"]
    all_data=[]

    for stock in stocks:
        dates = pd.date_range(start="2025-01-01", periods=100)
        data = {
        "Date": dates,
        "Stock":[stock]*100,
        "Open": np.random.uniform(100, 200, 100),
        "High": np.random.uniform(100, 200, 100),
        "Low": np.random.uniform(90, 195, 100),
        "Close": np.random.uniform(100, 200, 100),
        "Volume": np.random.randint(100000, 1000000, 100)
    }
        all_data.append(pd.DataFrame(data))
    df = pd.concat(all_data)
    df.to_csv("stock_data.csv", index=False)
    return df

try:
    df_stock = pd.read_csv("stock_data.csv")
except FileNotFoundError:
    df_stock = create_dummy_stock_data()

# -----------------------
# 2️⃣ Train RandomForest Model
# -----------------------
df_stock['Price_Up'] = (df_stock['Close'].shift(-1) > df_stock['Close']).astype(int)
df_stock = df_stock.dropna(subset=['Price_Up'])

X = df_stock[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df_stock['Price_Up']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "stock_model_rf.pkl")
joblib.dump(X.columns.tolist(), "stock_features.pkl")
features = X.columns.tolist()

# -----------------------
# 3️⃣ Database Setup
# -----------------------
conn = sqlite3.connect("stock_history.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS stock_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_symbol TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    price_up INTEGER
)
""")
conn.commit()

# -----------------------
# AUTO INSERT DATA INTO DB (IMPORTANT FIX)
# -----------------------
if pd.read_sql("SELECT COUNT(*) as count FROM stock_history", conn)["count"][0] == 0:
    temp_df = df_stock.copy()

    temp_df.rename(columns={
        "Stock": "stock_symbol",
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Price_Up": "price_up"
    }, inplace=True)

    temp_df.to_sql("stock_history", conn, if_exists="append", index=False)

# -----------------------
# 4️⃣ Streamlit App
# -----------------------
st.set_page_config(page_title="Stock Market Prediction", layout="wide")

# Beautiful Header
st.markdown("""
<div style='background-color:#0f4c81; padding:20px; border-radius:10px; text-align:center; color:white; margin-bottom:20px;'>
    <h1>Stock Market Prediction App</h1>
    <p>Predict if a stock price will go UP or DOWN</p>
</div>
""", unsafe_allow_html=True)


# Sidebar for historical data filters
st.sidebar.header(" Filter Historical Data")

symbols_db = pd.read_sql(
    "SELECT DISTINCT stock_symbol FROM stock_history", conn
)["stock_symbol"].dropna().tolist()

default_symbols = ["AAPL", "TSLA", "GOOG", "MSFT"]

all_symbols = sorted(set(default_symbols + symbols_db))

selected_symbol = st.sidebar.selectbox(
    "Select Stock Symbol",
    ["All"] + all_symbols,
    key="sidebar_stock"
)

start_date, end_date = st.sidebar.date_input("Select Date Range",
                                             value=[datetime(2025, 1, 1), datetime.now()])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Bulk CSV Upload", "Single Stock Input", "Historical Data","AI Chatbot"])

# -----------------------
# TAB 1: Bulk CSV Upload
# -----------------------
with tab1:
    st.subheader(" Upload Stock CSV")
    uploaded_file = st.file_uploader("Upload CSV with stock data", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        for col in features:
            if col not in df.columns:
                df[col] = 0

        df_encoded = df[features]
        predictions = model.predict(df_encoded)
        df["Price Up (1=Yes)"] = predictions

        st.success("Predictions generated successfully!")
        st.dataframe(df)
         # ✅ PUT DOWNLOAD BUTTON HERE
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="stock_data.csv",
            mime="text/csv"
        )

        # Metric cards
        total_up = int(df["Price Up (1=Yes)"].sum())
        total_down = len(df) - total_up
        col1, col2 = st.columns(2)
        col1.metric("📈 Total UP Predictions", total_up)
        col2.metric("📉 Total DOWN Predictions", total_down)

        # Interactive pie chart
        fig = px.pie(df, names="Price Up (1=Yes)", title="Prediction Summary",
                     color_discrete_sequence=['#eb5757','#6fcf97'])
        st.plotly_chart(fig)

        st.download_button("Download Predictions CSV",
                           df.to_csv(index=False), "predicted_stock_results.csv", mime="text/csv")

# -----------------------
# TAB 2: Single Stock Input
# -----------------------
with tab2:
    st.subheader(" Manual Prediction")
    col1, col2 = st.columns([1,1])

    with col1:
        stock_symbol = st.selectbox("Select Stock", ["AAPL","TSLA","GOOG","MSFT"])
        df_stock_filtered = df_stock[df_stock["Stock"] == stock_symbol]
        open_price = st.number_input("Open Price", 0.0)
        high_price = st.number_input("High Price", 0.0)
        low_price = st.number_input("Low Price", 0.0)
        close_price = st.number_input("Close Price", 0.0)
        volume = st.number_input("Volume", 0)

        predict = st.button("Predict")
    with col2:
        st.empty()
    if predict:
        input_dict = {
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        }

        input_df = pd.DataFrame([input_dict])[features]
        prediction = model.predict(input_df)[0]
        st.session_state["last_prediction"] = {
                "input": input_dict,
                "result": "Price will go UP 📈" if prediction == 1 else "Price will go DOWN 📉"
                }
        result_text = "Price will go UP 📈" if prediction == 1 else "Price will go DOWN 📉"
        result_color = "green" if prediction == 1 else "red"
        st.markdown(f"<div style='background-color:{'#6fcf97' if prediction==1 else '#eb5757'}; padding:20px; border-radius:10px; text-align:center;'>"
                    f"<h2 style='color:white'>{result_text}</h2></div>", unsafe_allow_html=True)

        # Bar chart of inputs
        chart_df = pd.DataFrame.from_dict(input_dict, orient="index", columns=["Value"])
        st.bar_chart(chart_df)

# -----------------------
# TAB 3: Historical Data
# -----------------------
with tab3:
    st.subheader("Historical Stock Data")

    # 1️⃣ Upload Historical Data
    st.markdown("### Upload Historical Stock Data")
    hist_file = st.file_uploader("Upload CSV with historical stock data", type="csv", key="hist_file_upload")
    
    if hist_file:
        hist_df = pd.read_csv(hist_file)
        required_cols = {"stock_symbol", "date", "open", "high", "low", "close", "volume", "price_up"}
        if not required_cols.issubset(hist_df.columns):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            hist_df.to_sql("stock_history", conn, if_exists="append", index=False)
            st.success("✅ Historical data uploaded successfully!")

    # 2️⃣ View & Filter Historical Data
    st.markdown("### View / Download Historical Data")
    symbols = pd.read_sql(
    "SELECT DISTINCT stock_symbol FROM stock_history", conn
)["stock_symbol"].dropna().tolist()

    selected_symbol_hist = st.selectbox(
    "Select Stock Symbol",
    ["All"] + symbols,
    key="history_stock_select"
)
    start_date, end_date = st.date_input(
    "Select Date Range",
    value=[datetime(2025,1,1), datetime.now()],
    key="history_date_range"
)
    query = "SELECT * FROM stock_history WHERE 1=1"
    params = []
    if selected_symbol_hist != "All":
       query += " AND stock_symbol = ?"
       params.append(selected_symbol_hist)
    if start_date and end_date:
        query += " AND date BETWEEN ? AND ?"
        params.extend([str(start_date), str(end_date)])

    df_hist = pd.read_sql(query, conn, params=params)
    st.dataframe(df_hist)

    # Download button
    csv_hist = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data", csv_hist, "historical_stock_data.csv", "text/csv")    # -----------------------


# -----------------------
# TAB 4: Chatbot
# -----------------------
with tab4:
    chatbot_ui()