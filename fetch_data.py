import yfinance as yf
import os
import pandas as pd

tickers = ["GOLD","AAPL", "TSLA","2330.TW""BTC-USD"]
start_date = "2015-01-01"
end_date = "2025-04-12"

if not os.listdir("data"):
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        filename = f"data/{ticker}.csv"
        data.to_csv(filename)
        print(f"儲存到：{filename}")