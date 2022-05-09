import pandas as pd
import numpy as np
from datetime import datetime
import backtrader as bt

# Load data
data = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/AAPL.csv',
                   index_col='Date')
df = data.iloc[-204:-1, :]  # all test data, test data length is 203
pred_y = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/pred_y.csv')
y = np.array(pred_y.iloc[:, -1])
df['trend'] = y
df.index = pd.to_datetime(df.index)
# print(df.head(20))


# Design strategy
class myStrategy(bt.Strategy):
    """
    Design strategy
    """
    params = dict(
    )

    def __init__(self):
        # open, close, predicted
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close
        self.data_trend = self.datas[0][-1]  # strategy signal

        # order/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

    # logging function
    def log(self, txt):
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED - -- Price: {order.executed.price: .2f}, '
                         f'Cost: {order.executed.value: .2f}, '
                         f'Commission: {order.executed.comm: .2f}')
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED - -- Price: {order.executed.price: .2f},'
                         f'Cost: {order.executed.value: .2f}, '
                         f'Commission: {order.executed.comm: .2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # Failed order
            self.log('Order Failed')
        self.order = None

    def next(self):
        # if there is order, won't create new order
        if self.order:
            return

        if not self.position:
            if self.data_trend > 0:
                # all-in
                print(self.broker.get_value())
                size = int(self.broker.get_value()/self.datas[0].open)
                # buy order
                self.log(f'BUY CREATED --- Size: {size}, '
                         f'Cash: {self.broker.getcash():.2f},'
                         f'Open: {self.data_open[0]}')
                self.order = self.buy(size=size)
        else:
            if self.data_trend == 0:
                # sell order
                self.log(f'SELL CREATED --- Size: {self.position.size}')
                self.order = self.sell(size=self.position.size)


cerebro = bt.Cerebro()
cerebro.broker.setcash(1000000.0)
cerebro.broker.setcommission(0.0002)
data = bt.feeds.PandasData(dataname=df, fromdate=datetime(2020, 3, 12), todate=datetime(2020, 12, 29),
                           timeframe=bt.TimeFrame.Days)
cerebro.adddata(data)
cerebro.addstrategy(myStrategy)
cerebro.broker = bt.brokers.BackBroker(slip_perc=0.002)
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='myPortfolio')
results = cerebro.run()

