import numpy as np
import pandas as pd
import pickle
from keras.layers import LSTM, Dense, Dropout
from keras import Sequential


data = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/AAPL.csv',
                   index_col='Date')

pre_length = 3
factor_num = 5


def model_lstm(factor_col):
    """

    :param factor_col: the feature column number, [0, 1, 2, 3, 4]
    :return: prediction of the feature value in the next day
    """
    train_x = np.ones((800, pre_length, 1))
    for i in range(pre_length, 800+pre_length):
        for j in range(pre_length):
            train_x[i-pre_length, j, 0] = data.iloc[i-pre_length+j, factor_col]
    train_y = data.iloc[pre_length:800+pre_length, factor_col]
    train_y = np.array(train_y)

    test_x = np.ones((data.shape[0]-800-pre_length, pre_length, 1))
    for i in range(800+pre_length, data.shape[0]):
        for j in range(pre_length):
            test_x[i-800-pre_length, j, 0] = np.array(data.iloc[i-pre_length+j, factor_col])

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(train_x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy', 'mse', 'mae'])
    model.fit(train_x, train_y, epochs=1, batch_size=64)
    pred_y = model.predict(test_x)
    # rmse = np.sqrt(np.mean(pred_y-test_y)**2)
    # print(rmse)

    return pred_y


# Predict the next value of each featue
forecast_y = pd.DataFrame()
for i in range(factor_num):
    forecast_y[str(i)] = list(model_lstm(i))
# print(forecast_y)

# Save forecast_y
data_output = open('/Users/wyb/PycharmProjects/Assignment3/database/forecast_y.pkl', 'wb')
pickle.dump(forecast_y, data_output)
data_output.close()



