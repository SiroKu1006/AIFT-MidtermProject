
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
