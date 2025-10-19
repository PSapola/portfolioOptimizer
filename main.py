from functions import *

optimal_weights = result.x

opt_return, opt_vol = portfolio_performance(weights=optimal_weights,mean_returns=mean_returns,cov_matrix=cov_matrix) # calculating the optimal returns and volatility and unpacking directly 
opt_sharpe = (opt_return-0.02)/opt_vol # new optimal sharpe with opt weights

print("=== OPTIMAL PORTFOLIO ===")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight*100:.2f}%")

print(f"\nExpected Annual Return: {opt_return*100:.2f}%")
print(f"Annual Volatility: {opt_vol*100:.2f}%")
print(f"Sharpe Ratio: {opt_sharpe:.3f}")