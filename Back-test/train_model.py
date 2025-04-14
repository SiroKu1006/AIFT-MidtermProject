
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# 讀取資料
df = pd.read_csv("data/AAPL.csv", skiprows=3, header=None)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
df = df.sort_index()

# 加入技術指標
df["SMA_5"] = df["Close"].rolling(window=5).mean()
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["Zscore_10"] = (df["Close"] - df["Close"].rolling(10).mean()) / df["Close"].rolling(10).std()
df["Return_1"] = df["Close"].pct_change(1)
df["Volatility_5"] = df["Return_1"].rolling(5).std()

# 建立 label：預測明天會不會漲
df["Label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# 清理缺失值
df = df.dropna()

# 特徵欄位
features = ["SMA_5", "SMA_10", "Zscore_10", "Return_1", "Volatility_5"]

# 切出 5 年訓練 + 5 年測試（2015~2019 訓練，2020~2024 測試）
train_df = df["2015":"2019"]
test_df = df["2020":"2024"]

X_train = train_df[features]
y_train = train_df["Label"]
X_test = test_df[features]
y_test = test_df["Label"]

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 建立與訓練模型
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 模型儲存
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/aapl_logistic_model.pkl")
joblib.dump(scaler, "model/aapl_scaler.pkl")

# 預測測試集
test_df.loc[:, "Predicted"] = model.predict(X_test_scaled)

# 根據預測建立持倉策略（1 表示買入，0 表示不持有）
test_df.loc[:, "Position"] = test_df["Predicted"].shift(1).fillna(0)
test_df.loc[:, "Return"] = test_df["Close"].pct_change()
test_df.loc[:, "Strategy Return"] = test_df["Position"] * test_df["Return"]

# 投資組合模擬
initial_capital = 10000
test_df.loc[:, "Portfolio Value"] = (1 + test_df["Strategy Return"]).cumprod() * initial_capital

# 繪圖與結果輸出
os.makedirs("result_ml", exist_ok=True)
plt.figure(figsize=(12,5))
plt.plot(test_df.index, test_df["Portfolio Value"], label="Strategy")
plt.title("AAPL ML-based Strategy Backtest (Logistic Regression)")
plt.ylabel("Portfolio Value ($)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("result_ml/aapl_ml_backtest.png")
plt.close()

# Summary 檔
total_return = (test_df["Portfolio Value"].iloc[-1] - initial_capital) / initial_capital
days = (test_df.index[-1] - test_df.index[0]).days
annual_return = (1 + total_return) ** (365 / days) - 1

with open("result_ml/aapl_ml_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"總報酬率：{total_return * 100:.2f}%\n")
    f.write(f"年化報酬率：{annual_return * 100:.2f}%\n")
    f.write(f"測試準確率：{accuracy_score(y_test, test_df['Predicted']) * 100:.2f}%\n")
