
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_stock_data(filepath):
    df = pd.read_csv(filepath, skiprows=3, header=None)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

def apply_dca_limited_strategy(df, total_capital=10000, monthly_invest=1000):
    df = df.copy()
    df = df.resample("M").first()
    df = df[df["Close"].notna()]
    
    max_months = total_capital // monthly_invest
    df = df.iloc[:max_months]

    df["Shares Bought"] = monthly_invest / df["Close"]
    df["Cumulative Shares"] = df["Shares Bought"].cumsum()
    df["Investment"] = monthly_invest * (df.index.to_series().rank(method="first").astype(int))
    df["Portfolio Value"] = df["Cumulative Shares"] * df["Close"]
    return df

def plot_dca(df, ticker):
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df["Portfolio Value"], label="Portfolio Value")
    plt.title(f"{ticker} - DCA (Limited Capital) Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"data/{ticker}_portfolio_dca_limited.png")
    plt.close()

def main(total_capital=10000, monthly_invest=50): #總金額10,000元，每月投資50元
    tickers = ["AAPL", "TSLA", "BTC-USD", "2330.TW", "GOLD"]
    os.makedirs("result", exist_ok=True)

    for ticker in tickers:
        filepath = f"data/{ticker}.csv"
        if not os.path.exists(filepath):
            print(f"找不到 {filepath}")
            continue

        print(f"\n處理中：{ticker}")
        df = load_stock_data(filepath)
        df_dca = apply_dca_limited_strategy(df, total_capital, monthly_invest)

        df_dca.to_csv(f"result/{ticker}_dca_limited.csv")
        plot_dca(df_dca, ticker)

        total_invested = df_dca["Investment"].iloc[-1]
        final_value = df_dca["Portfolio Value"].iloc[-1]
        total_return = (final_value - total_invested) / total_invested
        days = (df_dca.index[-1] - df_dca.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1

        with open(f"result/{ticker}_dca_limited_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"{ticker} 累積投入：${total_invested:.2f}\n")
            f.write(f"{ticker} 最終價值：${final_value:.2f}\n")
            f.write(f"{ticker} 總報酬率：{total_return * 100:.2f}%\n")
            f.write(f"{ticker} 年化報酬率：{annual_return * 100:.2f}%\n")

        print(f"{ticker} 年化報酬率：{annual_return * 100:.2f}%")

if __name__ == "__main__":
    main()
