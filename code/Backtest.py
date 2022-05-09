import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/AAPL.csv',
                   index_col='Date')
df = data.iloc[-204:-1, :]  # all test data, test data length is 203
pred_y = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/pred_y.csv')
y = np.array(pred_y.iloc[:, -1])
df['trend'] = y
df.index = pd.to_datetime(df.index)
# print(df.head(20))


# Back-test
class Backtest:

    def __init__(self, dataset):
        self.ret = []
        self.cumulative_ret = [1]
        self.position = 0
        self.cost = 0
        self.payoff = 0
        self.close = list(dataset['Close'])
        self.signal = pd.DataFrame(dataset['trend'])
        self.test_length = dataset.shape[0]

    def trade(self):
        for i in range(0, self.test_length-1):
            if self.signal.iloc[i, 0] == 1 and self.position == 0:
                print(str(self.signal.index[i]) + ' Buy order')
                print('Exe price: ' + str(self.close[i]))
                self.position = 1
            elif self.signal.iloc[i, 0] == 0 and self.position == 1:
                print(str(self.signal.index[i]) + ' Sell order')
                print('Exe price: ' + str(self.close[i]))
                self.position = 0
            if self.position == 1:
                self.ret.append(self.close[i+1]/self.close[i]-1)
                self.cumulative_ret.append(self.cumulative_ret[-1] * (1+self.ret[-1]))
            else:
                self.ret.append(0)
                self.cumulative_ret.append(self.cumulative_ret[-1] * 1)

    def annulized(self):
        a = self.cumulative_ret[-1]
        print(a)
        annualized_ret = a**(252/self.test_length)
        print('Annualized return is ' + str(annualized_ret))

    def sharpe(self):
        mu = np.mean(self.ret)*252/(self.test_length-1)
        sigma = np.std(self.ret)*np.sqrt(252/self.test_length-1)
        print('Sharpe ratio is ' + str(mu/sigma))

    def maxdrawdown(self):
        mdd = -1
        for i in range(len(self.cumulative_ret)-1):
            for j in range(i+1, len(self.cumulative_ret)):
                if self.cumulative_ret[j] < self.cumulative_ret[i]:
                    mdd = max(mdd, 1-self.cumulative_ret[j]/self.cumulative_ret[i])
                else:
                    break
        print('Max Drawdown is ' + str(mdd))

    def run(self):
        Backtest.trade(self)
        Backtest.annulized(self)
        Backtest.sharpe(self)
        Backtest.maxdrawdown(self)


bt = Backtest(df)
bt.run()

