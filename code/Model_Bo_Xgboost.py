import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import BayesianOptimization


data = pd.read_csv('/Users/wyb/PycharmProjects/Assignment3/database/AAPL.csv',
                   index_col='Date')
train_x = data.iloc[:800, :-1]
train_y = data.iloc[:800, -1]
model = xgb.Booster(params)

# bo = BayesianOptimization(
#     stock_xgb,
#     {'n_estimators': (10, 15),
#      'max_depth': (5, 10),
#      'learning_rate': (0.01, 0.02),
#      'min_child_weight': (1, 2),
#      'reg_alpha': (1, 2),
#      'reg_lamda': (1, 2),
#      'colsample_bytree': (1, 2),
#      'min_child_sample': (1, 2)
#      }
# )
# bo.maximize(iteration=25)