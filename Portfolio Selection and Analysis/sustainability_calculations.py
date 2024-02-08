#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


def calculate_sustainability_weight(optimal_weights, tickers, sustainable_stocks, unsustainable_stocks):
    """
    Calculate the total weights of the sustainable and unsustainable parts of the portfolio.

    Parameters:
    - optimal_weights (array): Optimal weights for the portfolio.
    - tickers (list): List of stock tickers corresponding to the optimal weights.
    - sustainable_stocks (list): List of stock tickers considered sustainable.
    - unsustainable_stocks (list): List of stock tickers considered unsustainable.

    Returns:
    - tuple: A tuple containing the total weight of sustainable stocks
    and the total weight of unsustainable stocks.
    """

    sustainable_weight = sum(weight for ticker, weight in zip(tickers, optimal_weights) if ticker in sustainable_stocks)
    unsustainable_weight = sum(weight for ticker, weight in zip(tickers, optimal_weights) if ticker in unsustainable_stocks)
    
    return sustainable_weight, unsustainable_weight


# In[3]:


def process_results_and_generate_summaries(optimal_results, tickers, sustainable_stocks, unsustainable_stocks, method_names):
    """
    Process optimization results and generate summaries for different methods.

    Parameters:
    - optimal_results (list of arrays): List of optimal weights for each method.
    - tickers (list): List of stock tickers corresponding to the optimal weights.
    - sustainable_stocks (list): List of stock tickers considered sustainable.
    - unsustainable_stocks (list): List of stock tickers considered unsustainable.
    - method_names (list): List of method names used for optimization.

    Returns:
    - summaries (list of str): List of summary texts for each optimization method.
    """

    fig, axs = plt.subplots(1, len(optimal_results), figsize=(15, 5))
    summaries = []  # List to hold all summary texts

    if len(optimal_results) == 1:  # Ensure axs is iterable for a single subplot
        axs = [axs]

    # Define pastel colors
    pastel_green = '#77dd77'  # Pastel green for sustainable
    pastel_red = '#ff6961'  # Pastel red for unsustainable

    for i, (optimal_weights, method_name) in enumerate(zip(optimal_results, method_names)):
        # Calculate sustainability weights
        sustainable_weight, unsustainable_weight = calculate_sustainability_weight(optimal_weights, tickers, sustainable_stocks, unsustainable_stocks)

        # Plotting
        axs[i].bar(['Sustainable', 'Unsustainable'], [sustainable_weight, unsustainable_weight], color=[pastel_green, pastel_red])
        axs[i].set_title(f'{method_name} Method')
        axs[i].set_ylabel('Weight')

        # Generate weight summary text
        weights_summary = f"{method_name} Method Sustainability Weights:\n" \
                          f"Sustainable Weight: {sustainable_weight:.4f}\n" \
                          f"Unsustainable Weight: {unsustainable_weight:.4f}\n"

        # Generate market preference summary text
        if sustainable_weight >= 1.5 * unsustainable_weight:
            preference_summary = f"According to the optimizing strategy '{method_name}', the sustainable companies outperform unsustainable ones in the chosen time frame."
        elif unsustainable_weight >= 1.5 * sustainable_weight:
            preference_summary = f"According to the optimizing strategy '{method_name}', the unsustainable companies outperform sustainable ones in the chosen time frame."
        else:
            preference_summary = f"According to the optimizing strategy '{method_name}', both types of companies perform relatively similarly."

        # Combine the summaries
        full_summary = weights_summary + preference_summary
        summaries.append(full_summary)  # Add the full summary to the list

        # Optionally, print the summary for immediate feedback
        print(full_summary + "\n---\n")

    plt.tight_layout()
    plt.show()

    return summaries  # Return the list of summary texts

