
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_stock_data(filepath):
    df = pd.read_csv(filepath, skiprows=3, header=None)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

def apply_lump_sum_strategy(df, initial_capital=10000):
    df = df.copy()
    buy_price = df["Close"].iloc[0]
    shares_bought = initial_capital / buy_price
    df["Shares Held"] = shares_bought
    df["Investment"] = initial_capital
    df["Portfolio Value"] = df["Close"] * shares_bought
    return df

def plot_lump_sum(df, ticker):
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df["Portfolio Value"], label="Portfolio Value")
    plt.title(f"{ticker} - Lump-Sum Investment Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"data/{ticker}_portfolio_lump_sum.png")
    plt.close()

def main(initial_capital=10000):
    tickers = ["AAPL", "TSLA", "BTC-USD", "2330.TW", "GOLD"]
    os.makedirs("result", exist_ok=True)

    for ticker in tickers:
        filepath = f"data/{ticker}.csv"
        if not os.path.exists(filepath):
            print(f"找不到 {filepath}")
            continue

        print(f"\n處理中：{ticker}")
        df = load_stock_data(filepath)
        df_lump = apply_lump_sum_strategy(df, initial_capital=initial_capital)

        df_lump.to_csv(f"result/{ticker}_lump_sum.csv")
        plot_lump_sum(df_lump, ticker)

        total_invested = df_lump["Investment"].iloc[0]
        final_value = df_lump["Portfolio Value"].iloc[-1]
        total_return = (final_value - total_invested) / total_invested
        days = (df_lump.index[-1] - df_lump.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1

        with open(f"result/{ticker}_lump_sum_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"{ticker} 初始投入：${total_invested:.2f}\n")
            f.write(f"{ticker} 最終價值：${final_value:.2f}\n")
            f.write(f"{ticker} 總報酬率：{total_return * 100:.2f}%\n")
            f.write(f"{ticker} 年化報酬率：{annual_return * 100:.2f}%\n")

        print(f"{ticker} 年化報酬率：{annual_return * 100:.2f}%")

if __name__ == "__main__":
    main()
