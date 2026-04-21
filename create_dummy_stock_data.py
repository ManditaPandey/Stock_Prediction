
import pandas as pd
import numpy as np

dates = pd.date_range(start="2025-01-01", periods=100)
data = {
    "Date": dates,
    "Stock": ["AAPL"]*100,
    "Open": np.random.uniform(100, 200, 100),
    "High": np.random.uniform(100, 200, 100),
    "Low": np.random.uniform(90, 195, 100),
    "Close": np.random.uniform(100, 200, 100),
    "Volume": np.random.randint(100000, 1000000, 100)
}

df = pd.DataFrame(data)
df.to_csv("stock_data.csv", index=False)
print("Dummy stock_data.csv created!")
