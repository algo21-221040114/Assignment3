import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization

pre_length = 3
factor_num = 5

data_input = open('C:/Users/221040114/Desktop/code/code/train_x.pkl', 'rb')
a = pickle.load(data_input)
train_x = np.ones((800, pre_length*factor_num))
for i in range(800):
    for j in range(pre_length):
        for k in range(factor_num):
            train_x[i, j * pre_length + k] = a[i, j, k]
# print(train_x.shape)

data_input = open('C:/Users/221040114/Desktop/code/code/train_y.pkl', 'rb')
train_y = pickle.load(data_input)
train_data = xgb.DMatrix(train_x, train_y)

data_input = open('C:/Users/221040114/Desktop/code/code/forecast_y.pkl', 'rb')
a = pickle.load(data_input)
test_x = xgb.DMatrix(a)

data_input = open('C:/Users/221040114/Desktop/code/code/test_y.pkl', 'rb')
test_y = pickle.load(data_input)

# # General XGBoost Model
# params_stock = {
#     'max_depth': 10,
#     'learning_rate': 0.1,
#     'min_child_weight': 3,
#     'reg_alpha': 2,
#     'colsample_bytree': 0.5,
#     'seed': 731
# }
#
# model_cv = xgb.cv(params_stock, dtrain=train_data, nfold=5, num_boost_round=100)
# print(model_cv)
# model_cv[['train-rmse-mean']].plot()
# model_cv[['test-rmse-mean']].plot()
# plt.show()


# Build target function
def model_cv(max_depth, learning_rate, min_child_weight, reg_alpha, colsample_bytree):

    params_stock = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'min_child_weight': min_child_weight,
        'reg_alpha': reg_alpha,
        'colsample_bytree': colsample_bytree,
        'seed': 0
    }

    model = xgb.cv(params=params_stock, dtrain=train_data, nfold=5)
    val_score = model['test-rmse-mean'].iloc[-1]
    return val_score*-1


# Bayesian Optimization
model_cv_bayes = BayesianOptimization(model_cv,
                                      {'max_depth': (5, 10),
                                       'learning_rate': (0.01, 0.1),
                                       'min_child_weight': (2, 3),
                                       'reg_alpha': (0.1, 0.2),
                                       'colsample_bytree': (0.7, 1)
                                       }
                                      )
model_cv_bayes.maximize(n_iter=25)
dict_params = model_cv_bayes.max['params']
dict_params['max_depth'] = int(dict_params['max_depth'])
best_model = xgb.train(dict_params, train_data, num_boost_round=100)
pred_y = best_model.predict(test_x)
print(pred_y)
