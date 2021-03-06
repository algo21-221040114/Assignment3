import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

# This model should run in Windows or Mac installed xgboost !!!

# Parameters
pre_length = 3
factor_num = 5


# Create train input for xgboost
data = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/AAPL.csv', index_col='Date')
train_x = np.ones((800, factor_num))
for i in range(800):
    for j in range(factor_num):
        train_x[i, j] = data.iloc[i+pre_length, j]
data_input = open('/Users/wyb/PycharmProjects/Assignment3/database/train_y.pkl', 'rb')
train_y = pickle.load(data_input)
train_data = xgb.DMatrix(train_x, train_y)

# Create test input for xgboost
data_input = open('/Users/wyb/PycharmProjects/Assignment3/database/forecast_y.pkl', 'rb')
a = pickle.load(data_input)
a = np.array(a).reshape((203, 5))  # 203 is the length of test data
test_x = a
data_input = open('/Users/wyb/PycharmProjects/Assignment3/database/test_y.pkl', 'rb')
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
def model_class(max_depth, learning_rate, min_child_weight, reg_alpha, colsample_bytree):

    model = XGBClassifier(max_depth=int(max_depth),  # 树的深度,最大的迭代次数
                          learning_rate=learning_rate,  # 学习率
                          min_child_weight=min_child_weight,  # 子节点中最小的样本权重和
                          reg_alpha=reg_alpha,  # L1正则项的权重，惩罚项，限制树节点数量，压制复杂度
                          colsample_bytree=colsample_bytree,  # 对特征采样的比例
                          objective='binary:logistic',  # 二分类的逻辑回归
                          seed=0,
                          use_label_encoder=False,
                          eval_metric='logloss')
    model.fit(train_x, train_y)
    prediction = model.predict(train_x)
    predictions = [round(value) for value in prediction]
    accuracy = accuracy_score(train_y, predictions)
    return accuracy


# Bayesian Optimization
model_class_bayes = BayesianOptimization(model_class,
                                         pbounds={'max_depth': (5, 10),
                                        'learning_rate': (0.01, 0.1),
                                        'min_child_weight': (2, 3),
                                        'reg_alpha': (0.1, 0.2),
                                        'colsample_bytree': (0.7, 1)
                                                  }
                                         )
model_class_bayes.maximize(n_iter=50)
dict_params = model_class_bayes.max['params']
dict_params['max_depth'] = int(dict_params['max_depth'])
print(dict_params)
values = []
for key in dict_params:
    values.append(dict_params[key])
best_model = XGBClassifier(colsample_bytree=values[0],
                           learning_rate=values[1],
                           max_depth=values[2],
                           min_child_weight=values[3],
                           reg_alpha=values[4],
                           use_label_encoder=False,
                           eval_metric='logloss')
best_model.fit(train_x, train_y)
pred_y = best_model.predict(test_x)
pred_y = pd.DataFrame(pred_y)
pred_y.to_csv('/Users/wyb/PycharmProjects/Assignment3/database/pred_y.csv')
# diff = (pred_y-test_y)**2
# print(np.sum(diff))