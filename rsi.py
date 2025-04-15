
import pandas as pd
import matplotlib.pyplot as plt
import os
import ta

def load_stock_data(filepath):
    df = pd.read_csv(filepath, skiprows=3, header=None)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

def apply_rsi_strategy(df, window=14, lower=30, upper=70):
    df = df.copy()
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    df["RSI"] = rsi
    df["Signal"] = 0
    df.loc[df["RSI"] < lower, "Signal"] = 1   # 超賣 → 買進
    df.loc[df["RSI"] > upper, "Signal"] = -1  # 超買 → 賣出
    df["Position"] = df["Signal"].shift(1)    # 信號延後一天執行
    return df.dropna()


def backtest_percent(df, initial_capital=10000, trade_percent=0.1):
    df = df.copy()
    cash = initial_capital
    shares = 0.0
    portfolio_values = []

    for date, row in df.iterrows():
        price = row["Close"]
        signal = row["Position"]

        if signal == 1:
            amount_to_invest = cash * trade_percent
            btc_to_buy = amount_to_invest / price
            if cash >= amount_to_invest:
                shares += btc_to_buy
                cash -= btc_to_buy * price

        elif signal == -1:
            btc_to_sell = shares * trade_percent
            shares -= btc_to_sell
            cash += btc_to_sell * price

        total_value = cash + shares * price
        portfolio_values.append((date, total_value))

    result_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index("Date")
    return result_df

def plot_rsi(df, ticker):
    plt.figure(figsize=(10, 4))
    plt.plot(df["RSI"], label="RSI", color="blue")
    plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    plt.title(f"{ticker} - RSI 指標")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"data/{ticker}_rsi_signal.png")
    plt.close()


def plot_portfolio(portfolio_df, ticker):
    plt.figure(figsize=(10,4))
    plt.plot(portfolio_df, label="Portfolio Value")
    plt.title(f"{ticker} - Backtest Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"data/{ticker}_portfolio_rsi.png")
    plt.close()


def main(trade_percent=0.01):
    tickers = ["AAPL", "TSLA", "BTC-USD", "2330.TW", "GOLD"]
    os.makedirs("result", exist_ok=True)

    for ticker in tickers:
        filepath = f"data/{ticker}.csv"
        if not os.path.exists(filepath):
            print(f"找不到 {filepath}")
            continue

        print(f"\n處理中：{ticker}")
        df = load_stock_data(filepath)

        df_rsi = apply_rsi_strategy(df)

        portfolio_df = backtest_percent(df_rsi, trade_percent=trade_percent)
        plot_rsi(df_rsi, ticker)
        plot_portfolio(portfolio_df, ticker)

        df_rsi.to_csv(f"result/{ticker}_signals_rsi.csv")
        portfolio_df.to_csv(f"result/{ticker}_portfolio_rsi.csv")

        total_return = (portfolio_df["Portfolio Value"].iloc[-1] - portfolio_df["Portfolio Value"].iloc[0]) / portfolio_df["Portfolio Value"].iloc[0] * 100
        max_drawdown = (portfolio_df["Portfolio Value"].cummax() - portfolio_df["Portfolio Value"]).max()

        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        annual_return = (1 + total_return / 100) ** (365 / days) - 1
        with open(f"result/{ticker}_summary_rsi.txt", "w", encoding="utf-8") as f:
            f.write(f"{ticker} RSI策略 總報酬率：{total_return:.2f}%\n")
            f.write(f"最大回落：${max_drawdown:.2f}\n")
            f.write(f"年化報酬率：{annual_return * 100:.2f}%\n")

        print(f"{ticker} RSI策略 總報酬率：{total_return:.2f}%")
        print(f"最大回落：${max_drawdown:.2f}")


if __name__ == "__main__":
    main()
