import numpy as np
import pandas as pd
import pickle
from keras.layers import LSTM, Dense, Dropout
from keras import Sequential


data = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/AAPL.csv',
                   index_col='Date')

pre_length = 1
factor_num = 5
train_x = np.ones((800, pre_length, factor_num))
for i in range(pre_length, 800+pre_length):
    x = np.ones((pre_length, factor_num))
    for j in range(factor_num):
        x[:, j] = data.iloc[i-pre_length:i, j]
    train_x[i-pre_length, :, :] = x
train_y = data.iloc[pre_length:800+pre_length, -1]
train_y = np.array(train_y)
data_output = open('/Users/wyb/PycharmProjects/Assignment3/database/train_x.pkl', 'wb')
pickle.dump(train_x, data_output)
data_output = open('/Users/wyb/PycharmProjects/Assignment3/database/train_y.pkl', 'wb')
pickle.dump(train_y, data_output)
data_output.close()

test_x = np.ones((data.shape[0]-800-pre_length, pre_length, factor_num))
for i in range(800+pre_length, data.shape[0]):
    x = np.ones((pre_length, factor_num))
    for j in range(factor_num):
        x[:, j] = data.iloc[i-pre_length:i, j]
    test_x[i-800-pre_length, :, :] = x
test_y = data.iloc[800+pre_length:data.shape[0], -1]
test_y = np.array(test_y)
data_output = open('/Users/wyb/PycharmProjects/Assignment3/database/test_x.pkl', 'wb')
pickle.dump(test_x, data_output)
data_output = open('/Users/wyb/PycharmProjects/Assignment3/database/test_y.pkl', 'wb')
pickle.dump(test_y, data_output)
data_output.close()

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(train_x.shape[1], factor_num)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy', 'mse', 'mae'])
model.fit(train_x, train_y, epochs=1, batch_size=64)
pred_y = model.predict(test_x)
rmse = np.sqrt(np.mean(pred_y-test_y)**2)
# print(rmse)

data_output = open('/Users/wyb/PycharmProjects/Assignment3/database/forecast_y.pkl', 'wb')
pickle.dump(pred_y, data_output)
data_output.close()



