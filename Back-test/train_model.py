
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from glob import glob

# 準備資料夾
os.makedirs("Back-test/model_1-2", exist_ok=True)
os.makedirs("Back-test/result_tv", exist_ok=True)
os.makedirs("Back-test/result_ml", exist_ok=True)

# 搜尋所有 data 底下的 .csv 檔
for filepath in glob("data/*.csv"):
    ticker = os.path.basename(filepath).replace(".csv", "")
    print(f"處理股票：{ticker}")

    # 讀取資料
    df = pd.read_csv(filepath, skiprows=3, header=None)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.sort_index()

    # 技術指標
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["Zscore_10"] = (df["Close"] - df["Close"].rolling(10).mean()) / df["Close"].rolling(10).std()
    df["Return_1"] = df["Close"].pct_change(1)
    df["Volatility_5"] = df["Return_1"].rolling(5).std()
    df["Label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    features = ["SMA_5", "SMA_10", "Zscore_10", "Return_1", "Volatility_5"]
    start_year, end_year = 2015, 2024

    for train in range(end_year - start_year):
        train_df = df[str(start_year):str(start_year + train)]
        test_df = df[str(start_year + train + 1):str(end_year)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df[features]
        y_train = train_df["Label"]
        X_test = test_df[features]
        y_test = test_df["Label"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        model_name = f"{ticker}_{start_year}_{start_year+train}"
        joblib.dump(model, f"Back-test/model_1-2/model_{model_name}.pkl")
        joblib.dump(scaler, f"Back-test/model_1-2/scaler_{model_name}.pkl")

        test_df = test_df.copy()
        test_df.loc[:, "Predicted"] = model.predict(X_test_scaled)
        test_df.loc[:, "Position"] = test_df["Predicted"].shift(1).fillna(0)
        test_df.loc[:, "Return"] = test_df["Close"].pct_change()
        test_df.loc[:, "Strategy Return"] = test_df["Position"] * test_df["Return"]
        test_df.loc[:, "Portfolio Value"] = (1 + test_df["Strategy Return"]).cumprod() * 10000

        total_return = (test_df["Portfolio Value"].iloc[-1] - 10000) / 10000
        days = (test_df.index[-1] - test_df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1

        # 存檔
        test_df.to_csv(f"Back-test/result_tv/portfolio_{model_name}.csv")
        plt.figure(figsize=(10, 4))
        plt.plot(test_df.index, test_df["Portfolio Value"], label="Portfolio")
        plt.title(f"{ticker} Portfolio {model_name}")
        plt.ylabel("Value ($)")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"Back-test/result_tv/portfolio_{model_name}.png")
        plt.close()

        with open(f"Back-test/result_ml/{ticker}_ml_summary.txt", "a", encoding="utf-8") as f:
            f.write(f"模型 {model_name}:")
            f.write(f"總報酬率：{total_return * 100:.2f}%\n")
            f.write(f"年化報酬率：{annual_return * 100:.2f}%\n")
            f.write(f"測試準確率：{accuracy_score(y_test, test_df['Predicted']) * 100:.2f}%\n\n")
