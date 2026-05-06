import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from textwrap import dedent
import yfinance as yf
from streamlit_autorefresh import st_autorefresh 


st_autorefresh(interval=60000, key="datarefresh")

st.set_page_config(page_title="STOCK MARKET PREDICTION", layout="wide")
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def rotating_globe():
    globe_html = """
    <html>
    <head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body style="margin:0; overflow:hidden; background:transparent;">
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);

        const renderer = new THREE.WebGLRenderer({alpha:true});
        const width = window.innerWidth;
        const height = 400;

        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        camera.updateProjectionMatrix();
        document.body.appendChild(renderer.domElement);

        const geometry = new THREE.SphereGeometry(2, 64, 64);
        

// 🌍 ADD HERE
const textureLoader = new THREE.TextureLoader();

const texture = textureLoader.load(
    'https://raw.githubusercontent.com/jeromeetienne/threex.planets/master/images/earthmap1k.jpg',
    () => {
        console.log("Earth texture loaded");
    },
    undefined,
    () => {
        console.log("Texture failed to load");
    }
);
        
        const baseMaterial = new THREE.MeshStandardMaterial({
    map: texture,
    
});
        const sphere = new THREE.Mesh(geometry, baseMaterial);
        scene.add(sphere);
        // 🌊 Ocean blue layer (ADD HERE)
const oceanMaterial = new THREE.MeshBasicMaterial({
    color: 0x0a1a3a,
    transparent: true,
    opacity: 0.7
});

const oceanSphere = new THREE.Mesh(
    new THREE.SphereGeometry(2.01, 64, 64),
    oceanMaterial
);

scene.add(oceanSphere);
        // ✨ Golden glow layer
const glowTexture = new THREE.TextureLoader().load(
    'https://raw.githubusercontent.com/jeromeetienne/threex.planets/master/images/earthmap1k.jpg'
);

const glowMaterial = new THREE.MeshBasicMaterial({
    map: glowTexture,
    color: 0xffd700,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending
});

const glowSphere = new THREE.Mesh(
    new THREE.SphereGeometry(2.02, 64, 64),
    glowMaterial
);

scene.add(glowSphere);
// 🌌 Atmosphere glow (ADD HERE)
const atmosphereMaterial = new THREE.MeshBasicMaterial({
    color: 0x4fa3ff,
    transparent: true,
    opacity: 0.08,
    blending: THREE.AdditiveBlending
});

const atmosphere = new THREE.Mesh(
    new THREE.SphereGeometry(2.2, 64, 64),
    atmosphereMaterial
);

scene.add(atmosphere);
        
        // 🌟 STAR FIELD (ADD HERE)
const starsGeometry = new THREE.BufferGeometry();
const starsMaterial = new THREE.PointsMaterial({ color: 0xffffff });

const starsVertices = [];
for (let i = 0; i < 5000; i++) {
    starsVertices.push(
        (Math.random() - 0.5) * 2000,
        (Math.random() - 0.5) * 2000,
        (Math.random() - 0.5) * 2000
    );
}

starsGeometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(starsVertices, 3)
);

const starField = new THREE.Points(starsGeometry, starsMaterial);
scene.add(starField);

        const light = new THREE.PointLight(0xffd700, 2, 100);
        light.position.set(5,5,5);
        scene.add(light);

        camera.position.z = 5;

        function animate() {
            requestAnimationFrame(animate);
            sphere.rotation.y += 0.002;
            oceanSphere.rotation.y += 0.002;
            glowSphere.rotation.y += 0.002;
            atmosphere.rotation.y += 0.001;
            starField.rotation.y += 0.0005; // ✨ motion
            renderer.render(scene, camera);
        }
        animate();
    </script>
    </body>
    </html>
    """
    
    st.components.v1.html(globe_html, height=400, width=1200)  

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
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
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
# AUTO INSERT DATA INTO DB
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


# Beautiful Header

st.markdown(dedent("""
<style>
@keyframes shine {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
</style>

<div style="padding:30px;border-radius:18px;text-align:center;margin-bottom:20px;
background:linear-gradient(180deg, rgba(0,0,0,0.6), rgba(0,198,255,0.05));
border:1px solid rgba(0,198,255,0.2);">

<div style="display:inline-block;padding:6px 14px;border-radius:999px;
background:rgba(0,198,255,0.12);color:#7dd3fc;font-size:13px;
letter-spacing:1px;margin-bottom:12px;border:1px solid rgba(0,198,255,0.25);">
AI POWERED MARKET INTELLIGENCE
</div>

<h1 style="font-size:48px;font-weight:800;margin:8px 0 12px 0;
background:linear-gradient(90deg,#00c6ff,#0072ff,#00ffd5,#00c6ff);
background-size:200% auto;
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
animation:shine 4s linear infinite;">
STOCK MARKET AI ASSISTANT
</h1>

<p style="font-size:16px;color:#cbd5e1;letter-spacing:0.6px;margin-bottom:18px;">
Predict trends | Analyze data | Make smarter decisions
</p>

</div>
"""), unsafe_allow_html=True)

st.markdown('<div style="width:100%; display:block;">', unsafe_allow_html=True)
rotating_globe()
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for historical data filters
st.sidebar.header(" FILTER HISTORICAL DATA")
conn.execute("DELETE FROM stock_history WHERE LOWER(stock_symbol) = 'stock'")
conn.commit()

symbols_db = pd.read_sql(
    "SELECT DISTINCT stock_symbol FROM stock_history", conn
)["stock_symbol"].dropna().tolist()

symbols_db = [
    s.strip().upper()
    for s in symbols_db
    if isinstance(s, str) and s.strip().lower() != "stock"
]

default_symbols = ["AAPL", "TSLA", "GOOG", "MSFT"]

all_symbols = sorted(set(default_symbols + symbols_db))

selected_symbol = st.sidebar.selectbox(
    "Select Stock Symbol",
    ["All"] + all_symbols,
    key="sidebar_stock"
)

date_range = st.sidebar.date_input(
    "SELECT DATE RANGE",
    value=(datetime(2025, 1, 1), datetime.now())
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]
# =========================
# SUPPORTED STOCK SYMBOLS GUIDE
# =========================
with st.sidebar.expander("WHAT STOCK SYMBOL WORKS?"):
    st.markdown("""
    **🇺🇸 US Stocks**
    - AAPL — Apple
    - TSLA — Tesla
    - MSFT — Microsoft
    - NVDA — Nvidia
    - AMZN — Amazon
    - META — Meta
    - GOOG — Google

    **🇮🇳 Indian NSE Stocks**  
    Add `.NS` at the end:

    - RELIANCE.NS — Reliance
    - TCS.NS — Tata Consultancy Services
    - INFY.NS — Infosys
    - HDFCBANK.NS — HDFC Bank
    - ICICIBANK.NS — ICICI Bank
    - SBIN.NS — State Bank of India
    - WIPRO.NS — Wipro
    - LT.NS — Larsen & Toubro

    **These will NOT work**
    - Meesho
    - Byju's
    - OYO
    - Any private/unlisted company

    **Tip:**  
    If an Indian stock does not work, try adding `.NS`.
    """)


# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Bulk CSV Upload",
    "Single Stock Input",
    "Historical Data",
    "AI Chatbot",
    "Live Market"
])

# -----------------------
# TAB 1: Bulk CSV Upload
# -----------------------
with tab1:
    st.subheader(" UPLOAD STOCK CSV")
    uploaded_file = st.file_uploader("Upload CSV with stock data", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        for col in features:
           if col not in df.columns:
            df[col] = 0

        df_encoded = df[features]
        predictions = model.predict(df_encoded)

        df["Price Up (1=Yes)"] = predictions
        df["Prediction Label"] = df["Price Up (1=Yes)"].map({
            1: "UP 📈",
            0: "DOWN 📉"
})

        st.success("PREDICTIONS GENERATED SUCESSFULLY!")
        st.dataframe(df)

# Download original predictions
        st.download_button(
        "DOWNLOAD CSV",
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

# Professional donut chart
        fig = px.pie(
        df,
        names="Prediction Label",
        color="Prediction Label",
        hole=0.45,
        title="AI PREDICTION DISTRIBUTION",
        color_discrete_map={
            "UP 📈": "#0ff439",
            "DOWN 📉": "#b20323"
    }
)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title=dict(x=0.5, font=dict(size=20)),
            legend=dict(
            orientation="h",
            y=-0.2,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)"
    ),
            margin=dict(t=60, b=70, l=20, r=20)
)

        fig.update_traces(
        textinfo="percent+label",
        marker=dict(line=dict(color="#111", width=2)),
        hovertemplate="<b>%{label}</b><br>Share: %{percent}<extra></extra>"
)

        st.markdown('<div class="chart-glass" style="position:relative; z-index:1;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.download_button(
        "Download Predictions CSV",
        df.to_csv(index=False),
        "predicted_stock_results.csv",
        mime="text/csv"
)
# -----------------------
# TAB 2: Single Stock Input
# -----------------------
with tab2:
    st.subheader("MANUAL PREDICTION")
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

    # ✅ EVERYTHING BELOW MUST BE INSIDE TAB2
    if predict:
        # -----------------------
        #  PREDICTION
        # -----------------------
        input_dict = {
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        }

        input_df = pd.DataFrame([input_dict])[features]
        prediction = model.predict(input_df)[0]

        result_text = "Price will go UP 📈" if prediction == 1 else "Price will go DOWN 📉"
        bg_color = "#00d057" if prediction == 1 else "#f70909"

        st.markdown(f"""
        <div style='background-color:{bg_color}; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='color:white'>{result_text}</h2>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------
        #  PRICE CHANGE + VOLATILITY
        # -----------------------
        change = close_price - open_price
        percent = (change / open_price) * 100 if open_price != 0 else 0

        colA, colB = st.columns(2)
        colA.metric("Price Change %", f"{percent:.2f}%")
        colB.metric("⚡ Volatility", f"{(high_price - low_price):.2f}")

        # -----------------------
        #  DONUT CHART
        # -----------------------
        import plotly.graph_objects as go

        buy_strength = max(percent, 0)
        sell_strength = abs(min(percent, 0))

        fig_donut = go.Figure(data=[go.Pie(
            labels=["Buy Pressure", "Sell Pressure"],
            values=[buy_strength + 0.01, sell_strength + 0.01],
            hole=0.6
        )])

        fig_donut.update_layout(
            title="Market Pressure",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------
        # 📈 TREND GRAPH
        # -----------------------
        hist_df = df_stock_filtered.tail(30)

        if not hist_df.empty:
            fig_freq = px.line(
                hist_df,
                x="Date",
                y="Close",
                title="Price Movement Trend (Last 30 Days)"
            )

            fig_freq.update_traces(mode="lines+markers")
            fig_freq.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
            st.plotly_chart(fig_freq, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------
        #  MOVING AVERAGE TREND
        # -----------------------
        df_stock_filtered['MA5'] = df_stock_filtered['Close'].rolling(5).mean()
        df_stock_filtered['MA20'] = df_stock_filtered['Close'].rolling(20).mean()

        if not df_stock_filtered.empty:
            if df_stock_filtered['MA5'].iloc[-1] > df_stock_filtered['MA20'].iloc[-1]:
                st.success("📈 Uptrend Detected (MA5 > MA20)")
            else:
                st.error("📉 Downtrend Detected (MA5 < MA20)")

        # -----------------------
        #  FINAL SIGNAL
        # -----------------------
        if prediction == 1 and percent > 0:
            st.success("Strong BUY Signal (AI + Trend)")
        elif prediction == 0 and percent < 0:
            st.error("Strong SELL Signal (AI + Trend)")
        else:
            st.warning(" Mixed Signals – Trade Carefully")

        # -----------------------
        # 📊 BAR CHART (INPUT FEATURES)
        # -----------------------
        chart_df = pd.DataFrame.from_dict(input_dict, orient="index", columns=["Value"])

        bar_fig = px.bar(
            chart_df,
            x=chart_df.index,
            y="Value",
            title="INPUT FEATURE DISTRIBUTION"
        )

        bar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# TAB 3: Historical Data
# -----------------------
with tab3:
    st.subheader("HISTORICAL STOCK DATA")

    # 1️⃣ Upload Historical Data
    st.markdown("### Upload Historical stock data")
    hist_file = st.file_uploader("Upload CSV with historical stock data", type="csv", key="hist_file_upload")
    
    if hist_file:
        hist_df = pd.read_csv(hist_file)
        required_cols = {"stock_symbol", "date", "open", "high", "low", "close", "volume", "price_up"}
        if not required_cols.issubset(hist_df.columns):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            hist_df.to_sql("stock_history", conn, if_exists="append", index=False)
            st.success(" Historical data uploaded successfully!")

    # 2️⃣ View & Filter Historical Data
    st.markdown("### View / DOWNLOAD HISTORICAL")
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
    if not df_hist.empty:
       line_fig = px.line(
        df_hist,
        x="date",
        y="close",
        color="stock_symbol",
        title="Stock Price Trend"
    )

    line_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
    st.plotly_chart(line_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Download button
    csv_hist = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data", csv_hist, "historical_stock_data.csv", "text/csv")    # -----------------------


# -----------------------
# TAB 4: Chatbot
# -----------------------
with tab4:
    st.subheader(" Market & Project Assistant")

    st.info("""
Try asking:
- What is stock prediction?
- What is OHLCV?
- Why prediction can be wrong?
- What is bullish?
- What is bearish?
- Explain this project
- What algorithm is used?
- What is Random Forest?
- What is Streamlit?
- What is SQLite?
- What is yfinance?
- Tell me viva answer
- What are limitations?
- Hello
""")

    user_query = st.text_input("Ask your question")

    if user_query:
        query = user_query.lower()

        # -----------------------
        # GREETINGS
        # -----------------------
        if "hello" in query or "hi" in query or "hey" in query:
            st.success("Hello! 👋 I am your Market Assistant. Ask me anything about stocks, prediction, ML, or this project.")

        elif "how are you" in query:
            st.success("I am doing great 😊 Ready to help you with stock market and project questions.")

        elif "who are you" in query:
            st.success("I am a stock market and project assistant built to answer questions without using any external API.")

        elif "help" in query:
            st.success("You can ask me about OHLCV, prediction, Random Forest, live market data, SQLite, Streamlit, yfinance, and viva questions.")

        elif "thank" in query:
            st.success("You're welcome 😊")

        elif "bye" in query:
            st.success("Goodbye! Keep learning and analyzing smartly 📊")

        # -----------------------
        # STOCK MARKET BASICS
        # -----------------------
        elif "stock market" in query:
            st.success("The stock market is a platform where shares of companies are bought and sold.")

        elif "stock" in query and "what" in query:
            st.success("A stock represents ownership in a company. Buying a stock means owning a small part of that company.")

        elif "share" in query:
            st.success("A share is a single unit of ownership in a company.")

        elif "company" in query and "listed" in query:
            st.success("A listed company is a company whose shares are available for trading on a stock exchange.")

        elif "nse" in query:
            st.success("NSE stands for National Stock Exchange of India.")

        elif "bse" in query:
            st.success("BSE stands for Bombay Stock Exchange, one of Asia’s oldest stock exchanges.")

        elif "ticker" in query or "symbol" in query:
            st.success("A stock symbol or ticker is a short code used to identify a company’s stock, like AAPL for Apple or TSLA for Tesla.")

        elif "private company" in query:
            st.success("Private companies are not publicly traded, so their stock data is usually not available on platforms like yfinance.")

        # -----------------------
        # OHLCV
        # -----------------------
        elif "ohlcv" in query:
            st.success("OHLCV stands for Open, High, Low, Close, and Volume. These are important stock market features used for analysis.")

        elif "open" in query:
            st.success("Open price is the price at which a stock starts trading during a selected period.")

        elif "high" in query:
            st.success("High price is the maximum price reached by a stock during a selected period.")

        elif "low" in query:
            st.success("Low price is the minimum price reached by a stock during a selected period.")

        elif "close" in query:
            st.success("Close price is the final trading price of a stock during a selected period.")

        elif "volume" in query:
            st.success("Volume shows the number of shares traded. High volume usually means strong market activity.")

        # -----------------------
        # TRENDS / SIGNALS
        # -----------------------
        elif "bullish" in query:
            st.success("Bullish means the stock or market is expected to move upward 📈")

        elif "bearish" in query:
            st.success("Bearish means the stock or market is expected to move downward 📉")

        elif "neutral" in query or "sideways" in query:
            st.success("Neutral or sideways market means price is not strongly moving up or down.")

        elif "trend" in query:
            st.success("Trend means the direction of price movement: upward, downward, or sideways.")

        elif "support" in query:
            st.success("Support is a price level where a stock often stops falling because buyers become active.")

        elif "resistance" in query:
            st.success("Resistance is a price level where a stock often struggles to move higher because sellers become active.")

        elif "volatility" in query:
            st.success("Volatility means how much the stock price fluctuates. High volatility means higher risk.")

        elif "buy signal" in query:
            st.success("A buy signal suggests positive movement, but it should not be considered guaranteed financial advice.")

        elif "sell signal" in query:
            st.success("A sell signal suggests possible weakness, but it should be verified using multiple indicators.")

        elif "risk" in query:
            st.success("Risk means the possibility of loss. Stock market investment always involves risk.")

        # -----------------------
        # PREDICTION RELATED
        # -----------------------
        elif "prediction" in query and "what" in query:
            st.success("Stock prediction means estimating whether a stock price may move up or down using historical or live data.")

        elif "how prediction" in query or "how does prediction" in query or "predict" in query:
            st.success("The system uses OHLCV values as input and predicts whether the price may move up or down.")

        elif "why prediction difficult" in query or "prediction difficult" in query:
            st.success("Stock prediction is difficult because prices depend on news, economy, company performance, global events, and investor sentiment.")

        elif "wrong prediction" in query or "prediction wrong" in query:
            st.success("Predictions can be wrong because the market is uncertain and sudden events cannot always be captured by historical data.")

        elif "accurate" in query or "accuracy" in query:
            st.success("Accuracy measures how many predictions are correct. In stock prediction, accuracy depends heavily on data quality.")

        elif "confidence" in query:
            st.success("Confidence represents how strongly the model supports its prediction. It can be calculated using probability scores from predict_proba().")

        elif "up" in query and "down" in query:
            st.success("UP means the next price is expected to be higher. DOWN means the next price is expected to be lower.")

        elif "real time prediction" in query:
            st.success("Real-time prediction uses recent live market data to generate current market signals.")

        elif "future price" in query:
            st.success("The system estimates direction, not exact future price. Predicting exact stock price is much harder.")

        # -----------------------
        # MACHINE LEARNING
        # -----------------------
        elif "machine learning" in query or "ml" in query:
            st.success("Machine Learning allows computers to learn patterns from data and make predictions.")

        elif "random forest" in query:
            st.success("Random Forest is an ensemble algorithm that combines many decision trees to make more stable predictions.")

        elif "why random forest" in query:
            st.success("Random Forest works well because it handles non-linear patterns and reduces overfitting compared to a single decision tree.")

        elif "decision tree" in query:
            st.success("A decision tree makes decisions by splitting data based on feature values.")

        elif "classification" in query:
            st.success("Classification means predicting a category. In this project, the categories are UP and DOWN.")

        elif "features" in query:
            st.success("Features are input columns used by the model. This project uses Open, High, Low, Close, and Volume.")

        elif "target" in query or "label" in query:
            st.success("The target variable is Price_Up, which shows whether the next closing price is higher than the current closing price.")

        elif "training" in query:
            st.success("Training means teaching the model using historical data so it can learn patterns.")

        elif "testing" in query:
            st.success("Testing means checking the model on unseen data to evaluate performance.")

        elif "train_test_split" in query:
            st.success("train_test_split divides data into training and testing parts.")

        elif "overfitting" in query:
            st.success("Overfitting happens when the model learns training data too well but performs poorly on new data.")

        elif "underfitting" in query:
            st.success("Underfitting happens when the model is too simple and fails to learn useful patterns.")

        # -----------------------
        # PROJECT SPECIFIC
        # -----------------------
        elif "explain this project" in query or "about project" in query or "project" in query:
            st.success("""
This project is a Stock Market AI Assistant.
It includes:
- CSV upload prediction
- Manual stock input prediction
- Historical stock data storage
- Live market watch
- Interactive charts
- Rule-based market chatbot
""")

        elif "objective" in query or "aim" in query:
            st.success("The objective is to help users analyze stock data, understand trends, and view prediction-based insights through an interactive dashboard.")

        elif "problem statement" in query:
            st.success("The problem is that stock market analysis is complex and time-consuming. This project simplifies analysis using prediction, visualization, and live market data.")

        elif "methodology" in query:
            st.success("Methodology: collect data → preprocess data → select OHLCV features → train model → predict movement → visualize results.")

        elif "workflow" in query or "pipeline" in query:
            st.success("Workflow: Data Collection → Data Cleaning → Feature Selection → Model Training → Prediction → Visualization → User Interaction.")

        elif "dataset" in query:
            st.success("The dataset contains stock values such as Date, Stock, Open, High, Low, Close, and Volume.")

        elif "csv" in query:
            st.success("CSV upload allows users to provide stock data and generate predictions in bulk.")

        elif "manual input" in query:
            st.success("Manual input allows the user to enter OHLCV values and get a prediction for one stock.")

        elif "historical data" in query:
            st.success("Historical data section stores and displays past stock records for analysis.")

        elif "live market" in query:
            st.success("Live market section shows current stock movement using data fetched from yfinance.")

        elif "chart" in query or "graph" in query:
            st.success("Charts help visualize price movement, prediction distribution, market pressure, and feature distribution.")

        elif "donut chart" in query:
            st.success("The donut chart shows buy and sell pressure visually.")

        elif "frequency polygon" in query:
            st.success("The frequency polygon shows stock price movement trend over recent data.")

        elif "database" in query or "sqlite" in query:
            st.success("SQLite is used to store historical stock data locally.")

        elif "streamlit" in query:
            st.success("Streamlit is used to create the interactive web dashboard using Python.")

        elif "plotly" in query:
            st.success("Plotly is used to build interactive charts and graphs.")

        elif "yfinance" in query:
            st.success("yfinance is used to fetch live and historical stock market data.")

        elif "joblib" in query:
            st.success("joblib is used to save trained model files and feature lists.")

        elif "limitation" in query:
            st.success("Limitations: predictions depend on data quality, stock market is uncertain, and sudden news cannot always be predicted.")

        elif "future scope" in query:
            st.success("Future scope includes sentiment analysis, news-based prediction, candlestick charts, portfolio tracking, alerts, and deep learning models.")

        elif "advantage" in query:
            st.success("Advantages: interactive UI, CSV upload, charts, live market data, historical storage, and simple prediction workflow.")

        elif "disadvantage" in query:
            st.success("Disadvantages: prediction reliability depends on dataset quality and market uncertainty.")

        # -----------------------
        # VIVA QUESTIONS
        # -----------------------
        elif "viva" in query:
            st.success("Viva answer: This project is an AI-based stock market assistant that uses OHLCV features to analyze and predict stock movement with visual insights.")

        elif "why this project" in query:
            st.success("This project is useful because stock analysis is difficult manually, and this dashboard makes analysis and visualization easier.")

        elif "why needed" in query:
            st.success("It is needed because users need a simple tool to understand stock trends, price movement, and market behavior.")

        elif "unique" in query:
            st.success("The unique part is the combination of prediction, live market tracking, historical data, charts, SQLite storage, and offline assistant.")

        elif "not financial advice" in query:
            st.success("This project is for educational and analytical purposes only. It should not be treated as financial advice.")

        # -----------------------
        # RANDOM GENERAL
        # -----------------------
        elif "best stock" in query:
            st.warning("I cannot guarantee the best stock. Stock selection requires research, risk analysis, and financial knowledge.")

        elif "should i buy" in query:
            st.warning("I cannot give financial advice. Please research carefully or consult a financial advisor.")

        elif "make me rich" in query:
            st.warning("I wish 😄 But stock market success needs knowledge, patience, discipline, and risk management.")

        elif "joke" in query:
            st.success("Why did the stock trader bring a ladder? Because they wanted to reach new highs 📈😄")

        else:
            st.warning("I can answer questions about stock market, prediction, ML, this project, or viva preparation.")

# -----------------------
# TAB 5: LIVE MARKET DATA
# -----------------------
with tab5:
    st.subheader("REAL-TIME MARKET WATCH")
    
    st.caption(" Updating every 1 min...")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    stock_list = ["AAPL", "TSLA", "GOOG", "MSFT", "RELIANCE.NS", "TCS.NS"]

    # =========================
    # TOP LIVE FLASHCARDS
    # =========================
    st.markdown("### LIVE STOCK MOVEMENT")

    cols = st.columns(3)

    for index, stock in enumerate(stock_list):
        data_card = yf.download(stock, period="5d", progress=False)

        if isinstance(data_card.columns, pd.MultiIndex):
            data_card.columns = data_card.columns.get_level_values(0)

        data_card = data_card.dropna()

        if not data_card.empty and len(data_card) > 1:
            latest_price = float(data_card["Close"].iloc[-1])
            prev_price = float(data_card["Close"].iloc[-2])
            change = latest_price - prev_price
            percent_change = (change / prev_price) * 100


            status = "UP " if change >= 0 else "DOWN "
            color = "#00ffd5" if change >= 0 else "#ff4d6d"
            with cols[index % 3]:
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    border: 1px solid {color};
                    border-radius: 18px;
                    padding: 18px;
                    margin-bottom: 16px;
                    box-shadow: 0 0 25px {color}55;
                    text-align: center;
                ">
                    <h3 style="color:white;">{stock}</h3>
                    <h2 style="color:{color};">${latest_price:.2f}</h2>
                    <p style="color:{color}; font-weight:700;">
                        {status}<br>
                        {change:+.2f} ({percent_change:+.2f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # ANY STOCK SEARCH + AI PREDICTION
    # =========================
    st.markdown("### SEARCH ANY STOCK + AI PREDICTIONS")

    user_stock = st.text_input(
        "Enter stock symbol",
        value="AAPL",
        placeholder="Example: AAPL, TSLA, MSFT, RELIANCE.NS, TCS.NS"
    ).upper().strip()

    period = st.selectbox(
        "Select Time Range",
        ["5d", "1mo", "6mo", "1y"],
        index=0
    )

    if user_stock:
        try:
            data = yf.download(user_stock, period=period, progress=False)

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.dropna()

            if data.empty:
                st.error("No data found. Please check the stock symbol.")
            elif len(data) < 2:
                st.warning("Not enough data available. Try 5d or 1mo.")
            else:
                latest_price = float(data["Close"].iloc[-1])
                prev_price = float(data["Close"].iloc[-2])

                change = latest_price - prev_price
                percent_change = (change / prev_price) * 100

                status = "UP " if change >= 0 else "DOWN "
                color = "#00ffd5" if change >= 0 else "#ff4d6d"

                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    border: 1px solid {color};
                    border-radius: 20px;
                    padding: 25px;
                    margin-bottom: 20px;
                    text-align: center;
                    box-shadow: 0 0 35px {color}66;
                ">
                    <h2 style="color:white;">{user_stock}</h2>
                    <h1 style="color:{color};">{status}</h1>
                    <h2 style="color:{color};">${latest_price:.2f}</h2>
                    <p style="color:{color}; font-size:18px; font-weight:700;">
                        {change:+.2f} ({percent_change:+.2f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # AI Prediction
                latest_row = data.iloc[-1]

                input_df = pd.DataFrame([{
                    "Open": latest_row["Open"],
                    "High": latest_row["High"],
                    "Low": latest_row["Low"],
                    "Close": latest_row["Close"],
                    "Volume": latest_row["Volume"]
                }])

                input_df = input_df[features]

                prediction = model.predict(input_df)[0]

                ai_result = "STOCK MAY GO UP" if prediction == 1 else "STOCK MAY GO DOWN "
                ai_color = "#00ffd5" if prediction == 1 else "#ff4d6d"

                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    border: 1px solid {ai_color};
                    border-radius: 20px;
                    padding: 22px;
                    margin-bottom: 25px;
                    text-align: center;
                    box-shadow: 0 0 30px {ai_color}55;
                ">
                    <h3 style="color:white;"> AI PREDICTION</h3>
                    <h1 style="color:{ai_color};">{ai_result}</h1>
                    <p style="color:#cbd5e1;">
                        Based on latest Open, High, Low, Close and Volume.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                fig = px.line(
                    data,
                    x=data.index,
                    y="Close",
                    title=f"{user_stock} Price Trend"
                )

                fig.update_traces(line=dict(color=color, width=3))

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    title=dict(x=0.5),
                    xaxis_title="Date",
                    yaxis_title="Price"
                )

                st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error("Something went wrong while fetching stock data.")
            st.write(e)