import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.optimize import minimize


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AVGO', 'UNH']
print('Downloading data...')
price_data = {}

for ticker in tickers:
    get_stock_data = yf.download(ticker,start='2020-01-01',end='2025-10-19',progress=False,auto_adjust=False)
    price_data[ticker] = get_stock_data['Close'].squeeze() #.squeeze() from DataFrame to series

data = pd.DataFrame(price_data)
print(data.head())

returns = data.pct_change().dropna() 

mean_returns = returns.mean() * 252 #252 trading days p.y.
cov_matrix = returns.cov() * 252 

def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights*mean_returns) #calculate portfolio returns
    portfolio_volatility = np.sqrt(np.dot(weights.T,np.dot(cov_matrix*weights))) #calculate portfolio volatility
    return portfolio_return, portfolio_volatility