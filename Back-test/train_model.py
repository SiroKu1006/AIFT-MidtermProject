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

# 設定初始訓練與測試時間範圍
start_year = 2015
end_year = 2024
train_years = 1  # 初始訓練資料為1年

os.makedirs("Back-test/model_1-2", exist_ok=True)
os.makedirs("Back-test/result_tv", exist_ok=True)
os.makedirs("Back-test/result_ml", exist_ok=True)

for train in range(end_year-start_year):

    train_df = df[str(start_year):str(start_year+train)]
    test_df = df[str(start_year+train+ 1):str(end_year)]

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

    # 儲存模型與標準化器
    model_filename = f"Back-test/model_1-2/model_{start_year}_{start_year+train}.pkl"
    scaler_filename = f"Back-test/model_1-2/scaler_{start_year}_{start_year+train}.pkl"
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    # 預測測試集
    test_df = test_df.copy()
    test_df.loc[:, "Predicted"] = model.predict(X_test_scaled)

    # 根據預測建立持倉策略（1 表示買入，0 表示不持有）
    test_df.loc[:, "Position"] = test_df["Predicted"].shift(1).fillna(0)
    test_df.loc[:, "Return"] = test_df["Close"].pct_change()
    test_df.loc[:, "Strategy Return"] = test_df["Position"] * test_df["Return"]

    # 投資組合模擬
    initial_capital = 10000
    test_df.loc[:, "Portfolio Value"] = (1 + test_df["Strategy Return"]).cumprod() * initial_capital
    
    # Summary 檔
    total_return = (test_df["Portfolio Value"].iloc[-1] - initial_capital) / initial_capital
    days = (test_df.index[-1] - test_df.index[0]).days
    annual_return = (1 + total_return) ** (365 / days) - 1
    
    model_name = f"{start_year}_{start_year+train}"
    # 儲存 portfolio 結果
    test_df.to_csv(f"Back-test/result_tv/portfolio_{model_name}.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(test_df.index, test_df["Portfolio Value"], label="Portfolio")
    plt.title(f"AAPL Portfolio {model_name}")
    plt.ylabel("Value ($)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"Back-test/result_tv/portfolio_{model_name}.png")
    plt.close()

    with open("Back-test/result_ml/aapl_ml_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"\n模型 {model_name}:\n")
        f.write(f"總報酬率：{total_return * 100:.2f}%\n")
        f.write(f"年化報酬率：{annual_return * 100:.2f}%\n")
        f.write(f"測試準確率：{accuracy_score(y_test, test_df['Predicted']) * 100:.2f}%\n")