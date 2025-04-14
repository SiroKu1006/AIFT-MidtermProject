
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_data(ticker):
    df = pd.read_csv(f"data/{ticker}.csv", skiprows=3, header=None)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df[["Close"]].rename(columns={"Close": ticker})

# 載入資料
aapl = load_data("AAPL")
tsla = load_data("TSLA")
df = pd.merge(aapl, tsla, left_index=True, right_index=True).dropna()


train = df["2015":"2022"].copy()
test = df["2023":"2024"].copy()

# 線性回歸估 β（AAPL ~ TSLA）
X_train = train[["TSLA"]].values
y_train = train["AAPL"].values
model = LinearRegression()
model.fit(X_train, y_train)
beta = model.coef_[0]

# spread & z-score（訓練集）
train["Spread"] = train["AAPL"] - beta * train["TSLA"]
spread_mean = train["Spread"].mean()
spread_std = train["Spread"].std()

# 測試集建構 z-score spread
test["Spread"] = test["AAPL"] - beta * test["TSLA"]
test["Zscore"] = (test["Spread"] - spread_mean) / spread_std

# 交易邏輯：Zscore > +1 賣AAPL買TSLA，Zscore < -1 買AAPL賣TSLA，Zscore ≈ 0 平倉
test["Signal"] = 0
test.loc[test["Zscore"] > 1, "Signal"] = -1
test.loc[test["Zscore"] < -1, "Signal"] = 1
test.loc[abs(test["Zscore"]) < 0.2, "Signal"] = 0  # 平倉訊號

# 訊號轉倉位（Position）
test["Position"] = test["Signal"].replace(to_replace=0, method="ffill").fillna(0)

# 計算報酬
test["AAPL Return"] = test["AAPL"].pct_change()
test["TSLA Return"] = test["TSLA"].pct_change()
test["Strategy Return"] = test["Position"] * (test["AAPL Return"] - beta * test["TSLA Return"])

# 投資組合績效
initial_capital = 10000
test["Portfolio Value"] = (1 + test["Strategy Return"]).cumprod() * initial_capital

# 輸出圖表與 summary
os.makedirs("result_ml", exist_ok=True)

plt.figure(figsize=(12,5))
plt.plot(test.index, test["Portfolio Value"], label="Pairs Trading Value (Z-score)")
plt.title("Pairs Trading Strategy: AAPL & TSLA (Z-score + Mean Reversion)")
plt.ylabel("Portfolio Value ($)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("result_ml/aapl_tsla_pairs_backtest_clean.png")
plt.close()

# summary
total_return = (test["Portfolio Value"].iloc[-1] - initial_capital) / initial_capital
days = (test.index[-1] - test.index[0]).days
annual_return = (1 + total_return) ** (365 / days) - 1

with open("result_ml/aapl_tsla_pairs_summary_clean.txt", "w", encoding="utf-8") as f:
    f.write(f"β (AAPL ~ TSLA)：{beta:.4f}\n")
    f.write(f"Spread Mean：{spread_mean:.4f}\n")
    f.write(f"Spread Std：{spread_std:.4f}\n")
    f.write(f"總報酬率：{total_return * 100:.2f}%\n")
    f.write(f"年化報酬率：{annual_return * 100:.2f}%\n")
