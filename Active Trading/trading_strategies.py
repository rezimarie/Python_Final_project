#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import math
from datetime import datetime
import datetime as dt
import yfinance as yf
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ## Baseline Strategies

# ### Random Buying 

# In[10]:


def random_trading(data: pd.DataFrame, initial_balance: float, description = True) -> float:
    """Takes in price data of an asset, performs the random buying and selling strategy
    for for all the days in the specified dataset, and returns the description of trades and final
    account balance"""
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    
    # Initialize variables for account balance and stock quantity held
    balance = initial_balance
    stock_quantity = 0

    # Define the trades for each day
    for day in range(1, len(data) + 1):
        action = random.choice(['buy', 'sell'])

        # Get the stock price for the current day
        current_price = data['Close'][day - 1]

        if action == 'buy':
            # Randomly determine the quantity to buy
            buy_quantity = random.randint(1, 10)

            total_cost = buy_quantity * current_price

            if total_cost <= balance:
                # Perform the purchase
                stock_quantity += buy_quantity
                balance -= total_cost
                
                # Describe trade if specified
                if description == True:
                    print(f"Day {day}: Bought {buy_quantity} stocks at ${current_price:.2f} each.")

        elif action == 'sell' and stock_quantity > 0:
            # Randomly determine the quantity to sell
            sell_quantity = random.randint(1, stock_quantity)

            total_earning = sell_quantity * current_price

            # Perform the sale
            stock_quantity -= sell_quantity
            balance += total_earning
            
            # Describe the trade if specified
            if description == True:
                print(f"Day {day}: Sold {sell_quantity} stocks at ${current_price:.2f} each.")

    # Sell remaining stocks on the last day
    if stock_quantity > 0:
        total_earning = stock_quantity * current_price
        balance += total_earning
        
        if description == True:
            print(f"Final day: Sold remaining {stock_quantity} stocks at ${current_price:.2f} each.")

    if description == True:
        print(f"Final balance: ${balance:.2f}")
    
    else:
        return balance


# ### Buy and Hold

# In[5]:


def buy_and_hold(data: pd.DataFrame, initial_balance: float, description = True) -> float:
    """Takes in price data of an asset and returns final balance after investing the initial_balance
     amount at the start of the time-series, and selling the position at the end of the time-series"""
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    
    # Initialize variables for account balance and stock quantity held
    stock_quantity = initial_balance / data["Adj Close"][0]
    
    # Final balance
    final_balance = stock_quantity * data["Adj Close"][-1]
    
    if description == True:
        print(f"Final balance: ${final_balance:.2f}")
    
    else:
        return final_balance


# ## Data-based Strategies

# ### Moving Average Crossover strategy

# In[6]:


# Simple moving average function
def SMA(data: pd.DataFrame, price_info: str = "Close", period: int = 30) -> float:
    """Takes in a dataframe with historical prices of an asset,
    type of price information used, such as Close or
    Adjusted close, and the rolling window period, and calculates
    the average"""

    # Check if the input dataframe is empty
    if data.empty:
        print("Input dataframe must be non-empty")
        return None

    # Check if the input data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        print("Data must be a Pandas DataFrame")
        return None

    # Check if the period is longer than the number of rows in the input dataframe
    if period > len(data):
        print("Moving average period must be shorter than the time frame of the dataset")
        return None

    # Check if the price_info is one of the expected values
    valid_price_info = ["Close", "Open", "Adj Close"]
    if price_info not in valid_price_info:
        print("price_info must be from the list", valid_price_info)
        return None

    return data[price_info].rolling(window=period).mean()


# In[7]:


def MA_strategy(data: pd.DataFrame, long_ma: int, short_ma: int) -> pd.DataFrame:
    """Takes in a time series dataframe with the asset information,
    produces buy and sell signals based on a crossover strategy
    with specified period from which to calculate the 
    short term and long term moving averages."""
    
    # Check if the input dataframe is empty
    if data.empty:
        print("Input dataframe must be non-empty")
        return None

    # Add the moving average columns to the dataset
    data["LongMA"] = SMA(data, "Open", period=long_ma)
    data["ShortMA"] = SMA(data, "Open", period=short_ma)
    
    # Check if the moving averages columns are empty
    if data["LongMA"].dropna().empty or data["ShortMA"].dropna().empty:
        print("Moving average calculation resulted in empty columns. Adjust the input parameters.")
        return None
    
    # Define buy and sell signals
    data["Signal"] = np.where(data["ShortMA"] > data["LongMA"], 1, 0)
    data["Position"] = data["Signal"].diff()
    data["Buy"] = np.where(data["Position"] == 1, data["Open"], np.NaN)
    data["Sell"] = np.where(data["Position"] == -1, data["Open"], np.NaN)
    
    return data


# ### XGBoost regression based strategy 

# In[8]:


def xgb_signals(data: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """Takes in a Pandas dataframe containing information on an asset
    from Yahoo Finance and the proportion of the dataset to be used for testing performance,
    and returns a dataframe with buy and sell signals for a specified period"""

    # Check if the input dataframe is empty
    if data.empty:
        print("Input dataframe must be non-empty")
        return None

    # Check if the test_size is valid
    if test_size <= 0 or test_size >= 1:
        print("Test size must be strictly greater than 0 and less than 1")
        return None

    # Creating a features dataset and target variable dataset
    features = data[["Open", "Volume"]]
    target = data["Close"]
    
    # Train test split
    feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=test_size, shuffle=False)
    
    # Check if the test set is empty
    if feature_test.empty or target_test.empty:
        print("Test set is empty. Adjust the test_size parameter or check the input data.")
        return None
      
    # Train the model
    model = xgb.XGBRegressor()
    model.fit(feature_train, target_train)
               
    # Predicted values of the model
    predictions = model.predict(feature_test)
               
    # Create a dataframe with lagged version of close prices
    lagged_actual_prices = target_test.shift(1)
    
    # Define signal conditions
    conditions = [
        predictions > lagged_actual_prices,
        predictions < lagged_actual_prices
    ]
    values = [1, -1]
    
    # Set up the trade signals DataFrame
    trade_signals_df = pd.DataFrame(index=lagged_actual_prices.index)
    
    # Include the trade signals in the dataframe
    trade_signals_df['Trade_Signal'] = np.select(conditions, values, default=0)
    
    # Filter only rows where Trade signal is not neutral
    trade_signals_df = trade_signals_df[trade_signals_df['Trade_Signal'] != 0]
    
    # Add the close prices for the days with trade signals
    trade_signals_df['Close_Price_On_Trade'] = target_test.loc[trade_signals_df.index]
    
    return trade_signals_df


# In[ ]:




