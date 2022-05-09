# Assignment3

# Introduction
This assignment refer to a paper, 'Forecast of LSTM-XGBoost in Stock Price Based on Optimization', published by Intelligent Automation & Soft Computing. 
You can access to the paper through https://www.techscience.com/iasc/v29n3/43035/pdf. 
The main proposal is to predict the up or down direction of stock price in the next day, with XGBoost. 
Two special ideas of this paper, one is to use Bayesian Optimization to search the optimal parameters of XGBoost model.
Another is to use LSTM as a pre-training model, the prediction price features of the next day is applied to be the input for XGBoost model.
All research result is based on the data of APPLE from 2017.01.01 to 2020.12.31, you can adjust parameters according to different stocks.

# Environment
Pycharm (Professional Edition) Python 3.7

# Requirements 
numpy 
pandas 
pickle
keras
xgboost
BayesianOptimization

As this strategy utilize the xgboost package, it's highly recommended conduct in Windows or Mac installed brew.

# Model Elaboration

# Back-test Result
