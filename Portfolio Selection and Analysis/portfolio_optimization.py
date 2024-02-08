#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import packages
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from scipy import optimize


# ## Auxiliary Functions

# In[7]:


# Portfolio standard deviation function
def standard_deviation(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Calculates the standard deviation of the portfolio"""
    
    # Check if weights is a NumPy array
    if not isinstance(weights, np.ndarray):
        raise ValueError("Weights must be a NumPy array")

    # Check if cov_matrix is a Pandas DataFrame
    if not isinstance(cov_matrix, pd.DataFrame):
        raise ValueError("Covariance matrix must be a Pandas DataFrame")

    # Check if DataFrame is not empty
    if cov_matrix.empty:
        raise ValueError("Input DataFrame must be non-empty")

    # Check for NaN or infinite values in DataFrame
    if np.isnan(weights).any() or cov_matrix.isnull().any().any():
        raise ValueError("Input arrays contain NaN or infinite values")

    # Check for positive definiteness of covariance matrix
    if not np.all(np.linalg.eigvals(cov_matrix) >= 0):
        raise ValueError("Covariance matrix must be positive semi-definite")

    # Check for shape compatibility
    if cov_matrix.shape[0] != cov_matrix.shape[1] or len(weights) != cov_matrix.shape[0]:
        raise ValueError("Incompatible dimensions for weights and covariance matrix")

    # Calculate standard deviation
    variance = weights @ cov_matrix @ weights
    return np.sqrt(variance)

# Negative Expected Return function
def neg_expected_return(weights: np.ndarray, log_returns: pd.Series) -> float:
    """Calculates the annualized expected return of the portfolio
    as a simple mean of the logarithmic returns on specified assets given NumPy
    array of weights along with specified asset returns in the portfolio"""
    
    # Check if weights is a NumPy array
    if not isinstance(weights, np.ndarray):
        raise ValueError("Weights must be a NumPy array")

    # check if log_returns is a pandas dataframe or series
    if not isinstance(log_returns, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a Pandas Series")

    # Check if Series is not empty
    if log_returns.empty:
        raise ValueError("Input Series must not be empty")

    # Check for NaN or infinite values in Series
    if np.isnan(weights).any() or log_returns.isnull().any().any():
        raise ValueError("Input arrays contain NaN values")

    # Calculate annualized expected return
    return -np.sum(log_returns.mean() * weights) * 252

# Negative Sharpe Ratio function
def neg_sharpe_ratio(weights: np.ndarray, log_returns: pd.Series,
                     cov_matrix: pd.DataFrame, risk_free_rate: float = 0.2) -> float:
    """Calculates the negative Sharpe ratio of the portfolio given weights, Pandas Series of logarithmic
    returns, portfolio covariance matrix, and the risk-free rate"""
    
    # Check if weights is a NumPy array
    if not isinstance(weights, np.ndarray):
        raise ValueError("Weights must be a NumPy array")

    # Check if log_returns is a Pandas Series and cov_matrix is a Pandas DataFrame
    if not isinstance(log_returns, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a Pandas Series or dataframe")
        
    # Check if cov_matrix is a Pandas DataFrame
    if not isinstance(cov_matrix, pd.DataFrame):
        raise ValueError("Inputs must be Pandas Series for log_returns and Pandas DataFrame for cov_matrix")

    # Check if Series or DataFrame are not empty
    if log_returns.empty or cov_matrix.empty:
        raise ValueError("Input Series or DataFrame must not be empty")

    # Check for NaN or infinite values in Series or DataFrame
    if np.isnan(weights).any() or log_returns.isnull().any().any() or cov_matrix.isnull().any().any():
        raise ValueError("Input arrays contain NaN values")

    # Check for positive definiteness of covariance matrix
    if not np.all(np.linalg.eigvals(cov_matrix) > 0):
        raise ValueError("Covariance matrix must be positive definite")

    # Calculate negative Sharpe ratio
    return (neg_expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)


# ## Portfolio Selection Optimization

# In[8]:


# Optimize Portfolio function
def optimize_portfolio(adj_close_data: pd.DataFrame, min_weight: float,
                       max_weight: float, method: str, rf_rate: float = 0.02):
    """Takes in data on adjusted close prices of specified assets, minimum and maximum
    proportion of one asset in the portfolio, preferred portfolio selection method, either min_variance,
    max_exp_return, or max_sharpe, and returns an object containing optimized weights of the portfolio."""
    
    # Check for empty DataFrame
    if adj_close_data.empty:
        raise ValueError("Input DataFrame 'adj_close_data' is empty.")

    # Check for non-numeric values in DataFrame
    if not pd.api.types.is_numeric_dtype(adj_close_data.dtypes.all()):
        raise ValueError("DataFrame 'adj_close_data' must contain only numeric values.")

    # Log returns extraction
    log_returns = np.log(adj_close_data / adj_close_data.shift(1)).dropna()

    # Check for NaN or infinite values in input data
    if log_returns.isnull().values.any():
        raise ValueError("Input data contains NaN values.")

    # Check for valid method
    valid_methods = ["min_variance", "max_exp_return", "max_sharpe"]
    if method not in valid_methods:
        raise ValueError(f"'{method}' is an invalid method. Choose from {valid_methods}.")

    # Annualized covariance matrix
    cov_matrix = log_returns.cov() * 252
    
    # calculate variables for optimization
    init_weights = np.array([1/log_returns.shape[1]]*log_returns.shape[1])
    df_std_dev = standard_deviation(init_weights, cov_matrix)
    df_exp_return = -neg_expected_return(init_weights, log_returns=log_returns)
    df_sharpe = neg_sharpe_ratio(init_weights, log_returns, cov_matrix, rf_rate)
    
    # Set optimization constraints
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(min_weight, max_weight) for _ in range(log_returns.shape[1])]
    
    if method == "max_sharpe":
        # minimize negative Sharpe ratio
        optimized_results = optimize.minimize(neg_sharpe_ratio, init_weights,
                            args=(log_returns, cov_matrix, rf_rate), method='SLSQP', constraints=constraints, bounds=bounds)
        
    elif method == "min_variance":
        optimized_results = optimize.minimize(
            standard_deviation, init_weights,
            args=(cov_matrix), method='SLSQP', constraints=constraints, bounds=bounds)
        
    elif method == "max_exp_return":
        optimized_results = optimize.minimize(neg_expected_return, init_weights,
                            args=(log_returns), method='SLSQP', constraints=constraints, bounds=bounds)

    return optimized_results


# In[ ]:




