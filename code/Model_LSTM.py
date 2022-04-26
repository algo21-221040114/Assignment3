import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras import Sequential


data = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/AAPL.csv',
                   index_col='Date')
pre_length = 3
train_x = []
test_x = []
for i in range(pre_length, 800+pre_length):
    train_x.append(data.iloc[i-pre_length:i, -2])
train_y = data.iloc[pre_length:800+pre_length, -1]
train_x = np.array(train_x)
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
train_y = np.array(train_y)
for i in range(800+pre_length, data.shape[0]):
    test_x.append(data.iloc[i-pre_length:i, -2])
test_y = data.iloc[800+pre_length:data.shape[0], -1]
test_x = np.array(test_x)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
test_y = np.array(test_y)


model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=1, batch_size=64)
pred_y = model.predict(test_x)
rmse = np.sqrt(np.mean(pred_y-test_y)**2)
print(rmse)

