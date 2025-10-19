import yfinance as yf
import numpy as np
import pandas as pd



def get_tickers():
    ticker_amount = int(input('Enter the amount of stocks your portfolio will have: '))
    tickers = []
    for _ in range(ticker_amount):
        user_ticker = input('Enter a ticker: ').upper().strip()
        tickers.append(user_ticker)
    return tickers


def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights*mean_returns) #calculate portfolio returns
    portfolio_volatility = np.sqrt(np.dot(weights.transpose(),np.dot(cov_matrix,weights))) #calculate portfolio volatility
    return portfolio_return, portfolio_volatility


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate = 0.02): 
    p_return, p_vol = portfolio_performance(weights, mean_returns, cov_matrix) 
    sharpe = (p_return-risk_free_rate)/p_vol #formula for sharpe ratio
    return -sharpe # scipy doesn't have maximize(), returning -sharpe to find the best portfolio

def generate_random_portfolios(num_portfolios,mean_returns, cov_matrix):
    results = np.zeros((3,num_portfolios))

    for i in range(num_portfolios):
        num_assets = len(mean_returns)
        weights = np.random.random(num_assets) # random weights
        weights = weights / np.sum(weights) #normalizes weights

        p_return, p_vol = portfolio_performance(weights,mean_returns,cov_matrix)

        sharpe = -negative_sharpe(weights,mean_returns,cov_matrix, risk_free_rate=0.02)

        results[0,i] = p_vol
        results[1,i] = p_return
        results[2,i] = sharpe
    return results
