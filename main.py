from functions import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt 


tickers = get_tickers()
price_data = {}

print('Downloading data...')

for ticker in tickers:
    get_stock_data = yf.download(ticker,start='2020-01-01',end='2025-10-19',progress=False,auto_adjust=False)
    price_data[ticker] = get_stock_data['Close'].squeeze() #.squeeze() from DataFrame to series

data = pd.DataFrame(price_data)
returns = data.pct_change().dropna() 


num_assets = len(tickers) 
initial_guess = num_assets * [1/num_assets] #initial weight guess, even split between the number of stocks
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1} # the sum of all constraints - 1 MUST equal 0, if not, we have "money" sitting uninvested 
bounds = tuple((0,0.33) for _ in range(num_assets)) #the potential weights for each stock. NO short selling (no negative weights)


mean_returns = returns.mean() * 252 #252 trading days p.y.
cov_matrix = returns.cov() * 252 


results = generate_random_portfolios(5000,mean_returns,cov_matrix) #not to be mixed with result which is used for the optimized portfolio
result = minimize(negative_sharpe,initial_guess,args=(mean_returns,cov_matrix),method='SLSQP',bounds=bounds,constraints=constraints) #calculate optimal weights and sharpe for those weights

optimal_weights = result.x

opt_return, opt_vol = portfolio_performance(weights=optimal_weights,mean_returns=mean_returns,cov_matrix=cov_matrix) # calculating the optimal returns and volatility and unpacking directly 
opt_sharpe = (opt_return-0.02)/opt_vol # new optimal sharpe with opt weights


volatlities = results[0,:] #collecting all of the volatilities from the random portfolios
returns = results[1,:] #collecting all of the returns data from the random portfolios
sharpes = results[2,:] #collecting all of the sharpe ratios from the random portfolios


print("=== OPTIMAL PORTFOLIO ===")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight*100:.2f}%")

print(f"\nExpected Annual Return: {opt_return*100:.2f}%")
print(f"Annual Volatility: {opt_vol*100:.2f}%")
print(f"Sharpe Ratio: {opt_sharpe:.3f}")
