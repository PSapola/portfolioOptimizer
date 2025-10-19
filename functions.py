import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.optimize import minimize


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AVGO', 'UNH']
price_data = {}

for ticker in tickers:
    get_stock_data = yf.download(ticker,start='2020-01-01',end='2025-10-19')
    price_data[ticker] = get_stock_data['Close']
