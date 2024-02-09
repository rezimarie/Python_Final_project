# Final_project
Hi, welcome to our final project,
we offer tools for both active or passive trading. We also offer some insight into performance of sustainable companies, compared to unsistainable.

## Passive Investing
If you  prefer passive investing, you can use our tool for **Portfolio creation and analysis**. Here you can input stocks (or assets) of your interest or choose some of the offered stocks, enter time period and amount you want to invest. The tool will provide you with optimal weights among these assets according to three different investing strategies. The tools also provies Monte Carlo simulations of future development of the porfolio. The output is also coompanied by multiple visualisations.

## Active Trading
If you prefer active trading, take a look at the **Trading Strategies** repository. The notebook **Active Trading** contains two baseline strategies along with simulations of actual trading, and two data-baseed strategies that provide trading signals based on two different models. The first offered strategy is a basic Moving Average approach based on the idea of mean reversion. Long-term and Short-term moving averages are calculated, which are then used to spot trend changes based. The recommendation is to make trades at the points when these moving averages cross. When the Long-term average crosses below the Short-term average, the signal is to sell, because the short-term trend is about the revert to the Long-term average. Similar reasoning applies to the buy signals. As for the XGBoost regression-based strategy, the model (provide link to the model) uses a gradient-boosting approach on the provided dataset to predict daily close prices. The user can choose the train-test split of the dataset. The buy and sell signals for this strategy are provided based on whether the predicted close price for the next day is lower or greater than today´s close price. Therefore it generates signals for each day, which provides opportunity for higher frequency trading than the MA strategy. The idea is for the user to be able to compare the performance of the MA and XGBoost-based trading strategies on a specific stock (or any ticker for which data is available on Yahoo Finance) in a specific period with just buying and holding until the end of the period, or randomly buying and selling. Based on the information provided by the analysis, the user can make more informed decisions about whether or not trading a specific ticker is desirable for them, and perhaps, if the strategies provided offer a good approach for the chosen product. Contrary to our primary intention, we decided not to use ARIMA or GARCH models to base trading strategies on.

## Data source
For our project, we took the advantage of the yfinance library, which uses publicly available Yahoo Finance API to download historical data on a plethora of financial products such as stocks or indices. Obtaining data in all notebooks is based on the yfinance.download function.

## Sustainable vs. unsustainable companies
We also specifically focus on energy companies in notebook **To sustain or not to sustain**, where we offer comparison of sustainable and unsustainable ones, this can help to indicate whether the market situation is in favor of sustainable stocks or not. We do this by using our portfolio creation strategies and than observing, how many of the sustainable stocks made it to the final portfolio.




