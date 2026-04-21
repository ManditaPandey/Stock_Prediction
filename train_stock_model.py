

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# -----------------------
# Load Stock Data
# -----------------------
# CSV should have columns: ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']
df = pd.read_csv("stock_data.csv")  

# -----------------------
# Feature Engineering
# -----------------------
# Predict next day's price movement: 1 = Up, 0 = Down
df['Price_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Optional: drop last row (NaN in target)
df = df.dropna(subset=['Price_Up'])

# Features: OHLCV
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Target
y = df['Price_Up']

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# -----------------------
# Train RandomForest Model
# -----------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------
# Save Model & Features
# -----------------------
joblib.dump(model, "stock_model_rf.pkl")
joblib.dump(X.columns.tolist(), "stock_features.pkl")

# -----------------------
# Print Accuracy (Optional)
# -----------------------
accuracy = model.score(X_test, y_test)
print(f"Model trained! Test Accuracy: {accuracy:.2f}")
print("Model saved as 'stock_model_rf.pkl' and feature list as 'stock_features.pkl'")
