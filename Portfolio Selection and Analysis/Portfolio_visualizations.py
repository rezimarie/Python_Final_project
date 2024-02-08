#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Import packages
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import portfolio_optimization as pto


# In[3]:


def optimal_weights_plot(adj_close_df: pd.DataFrame, tickers: list[str]):
    """Takes in a Pandas dataframe with price information on assets in specified portfolio of tickers
    and returns a barplot of weights for each portfolio selection method"""
    
    # Auxiliary variables for analysis
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252

    # Optimal portfolio information for max_exp_return
    optimal_result_max_exp_return = pto.optimize_portfolio(adj_close_df, min_weight=0.0, max_weight=0.2, method="max_exp_return")
    optimal_weights_max_exp_return = optimal_result_max_exp_return.x

    # Optimal portfolio information for min_variance
    optimal_result_min_variance = pto.optimize_portfolio(adj_close_df, min_weight=0.0, max_weight=0.2, method="min_variance")
    optimal_weights_min_variance = optimal_result_min_variance.x

    # Optimal portfolio information for max_sharpe
    optimal_result_max_sharpe = pto.optimize_portfolio(adj_close_df, min_weight=0.0, max_weight=0.2, method="max_sharpe")
    optimal_weights_max_sharpe = optimal_result_max_sharpe.x

    # variance, expected return and Sharpe Ratio for each method

    # Max. Expected return
    max_exp_return = -pto.neg_expected_return(optimal_weights_max_exp_return, log_returns=log_returns)
    max_exp_volatility = pto.standard_deviation(optimal_weights_max_exp_return, cov_matrix)
    max_exp_sharpe = (-1) * pto.neg_sharpe_ratio(optimal_weights_max_exp_return, log_returns, cov_matrix, 0.02)

    # Min. Variance
    min_vol_return = -pto.neg_expected_return(optimal_weights_min_variance, log_returns=log_returns)
    min_vol_volatility = pto.standard_deviation(optimal_weights_min_variance, cov_matrix)
    min_vol_sharpe = (-1) * pto.neg_sharpe_ratio(optimal_weights_min_variance, log_returns, cov_matrix, 0.02)

    # Max. Sharpe Ratio
    max_sharpe_return = -pto.neg_expected_return(optimal_weights_max_sharpe, log_returns=log_returns)
    max_sharpe_vol = pto.standard_deviation(optimal_weights_max_sharpe, cov_matrix)
    max_sharpe_sharpe = (-1) * pto.neg_sharpe_ratio(optimal_weights_max_sharpe, log_returns, cov_matrix, 0.02)

    # show the portfolio information for each:
    print("Maxmimum Expected Return Method:")
    print(f"Expected Annual Return: {max_exp_return:.4f}")
    print(f"Expected Volatility: {max_exp_volatility:.4f}")
    print(f"Sharpe Ratio: {max_exp_sharpe:.4f}")

    print("Minimum Variance Method:")
    print(f"Expected Annual Return: {min_vol_return:.4f}")
    print(f"Expected Volatility: {min_vol_volatility:.4f}")
    print(f"Sharpe Ratio: {min_vol_sharpe:.4f}")

    print("Maximum Sharpe Ratio Method:")
    print(f"Expected Annual Return: {max_sharpe_return:.4f}")
    print(f"Expected Volatility: {max_sharpe_vol:.4f}")
    print(f"Sharpe Ratio: {max_sharpe_sharpe:.4f}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot for max_exp_return
    axs[0].bar(tickers, optimal_weights_max_exp_return)
    axs[0].set_title('Max Expected Return')
    axs[0].tick_params(axis='x', rotation=90)

    # Plot for min_variance
    axs[1].bar(tickers, optimal_weights_min_variance)
    axs[1].set_title('Min Variance')
    axs[1].tick_params(axis='x', rotation=90)

    # Plot for max_sharpe
    axs[2].bar(tickers, optimal_weights_max_sharpe)
    axs[2].set_title('Max Sharpe Ratio')
    axs[2].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


# In[4]:


def portfolio_breakdown_plot(adj_close_df, invested_amount, tickers):
    """Takes in a Pandas dataframe with price information on assets in specified portfolio of tickers
    and returns a barplot with a breakdown of the portfolio for each type of selection method"""
    
    # Auxiliary variables for analysis
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252

    # Optimal portfolio information for max_exp_return
    optimal_result_max_exp_return = pto.optimize_portfolio(adj_close_df, min_weight=0.0, max_weight=0.2, method="max_exp_return")

    # Optimal portfolio information for min_variance
    optimal_result_min_variance = pto.optimize_portfolio(adj_close_df, min_weight=0.0, max_weight=0.2, method="min_variance")

    # Optimal portfolio information for max_sharpe
    optimal_result_max_sharpe = pto.optimize_portfolio(adj_close_df, min_weight=0.0, max_weight=0.2, method="max_sharpe")

    # Portfolio dataframe for max_exp
    positions_max_exp = np.round(optimal_result_max_exp_return.x * float(invested_amount))
    portfolio_max_exp = pd.DataFrame({'Asset': tickers, 'Amount in USD': positions_max_exp})

    # Portfolio dataframe for min_variance
    positions_min_variance = np.round(optimal_result_min_variance.x * float(invested_amount))
    portfolio_min_variance = pd.DataFrame({'Asset': tickers, 'Amount in USD': positions_min_variance})

    # Portfolio dataframe for max_sharpe
    positions_max_sharpe = np.round(optimal_result_max_sharpe.x * float(invested_amount))
    portfolio_max_sharpe = pd.DataFrame({'Asset': tickers, 'Amount in USD': positions_max_sharpe})

    # Displaying portfolio breakdowns side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plot for max_exp_return
    portfolio_max_exp.plot(x='Asset', y='Amount in USD', kind='bar', color='skyblue', ax=axs[0])
    axs[0].set_title('Max Expected Return')

    # Plot for min_variance
    portfolio_min_variance.plot(x='Asset', y='Amount in USD', kind='bar', color='skyblue', ax=axs[1])
    axs[1].set_title('Min Variance')

    # Plot for max_sharpe
    portfolio_max_sharpe.plot(x='Asset', y='Amount in USD', kind='bar', color='skyblue', ax=axs[2])
    axs[2].set_title('Max Sharpe Ratio')

    # Adjust layout
    plt.tight_layout()

    plt.show()

