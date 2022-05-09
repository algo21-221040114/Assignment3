import pandas as pd
from pandas_datareader import data
from datetime import datetime


#  Read data
stock_name = 'AAPL'
start = datetime(2017, 1, 1)
end = datetime(2020, 12, 31)
df = data.DataReader(stock_name, 'yahoo', start, end)
trend = []

# Create label
for i in range(0, df.shape[0]):
    if i == df.shape[0]-1:
        trend.append(1)
    elif df.iloc[i, -1] < df.iloc[i+1, -1]:
        trend.append(1)
    else:
        trend.append(0)
df['trend'] = trend
df = df.drop(df.tail(1).index)
df = df.drop(['Volume'], axis=1)

# Save data
df.to_csv('/Users/wyb/PycharmProjects/Assignment3/database/AAPL.csv', sep=',', header=True, index=True)