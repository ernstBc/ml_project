from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

models={
    'LinearRegression':LinearRegression(),
    'DecisionTreeRegressor':DecisionTreeRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor':GradientBoostingRegressor(),
    'XGBRegressor': XGBRegressor(),
    'CatBoostRegressor': CatBoostRegressor(verbose=0)
}

models_params={
    'LinearRegression':{
        'fit_intercept':[True, False]
    },

    'DecisionTreeRegressor':{
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ["best", "random"],
        'max_depth':[1, 3 , 5, 10, None],
        'min_samples_split':[2, 5, 10, 20, None],
        'min_samples_leaf':[1, 5, 10, 20, None]
    },

    'KNeighborsRegressor': {
        'n_neighbors': [1, 5, 10, 15],
        'weights': ['uniform', 'distance']
    },

    'RandomForestRegressor': {
        'n_jobs':[-1],
        'n_estimators': [10, 50, 100, 500],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'max_depth':[1, 3 , 5, 10, None],
        'min_samples_split':[2, 5, 10, 20, None],
        'min_samples_leaf':[1, 5, 10, 20, None]

    },

    'GradientBoostingRegressor':{
        'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
        'learning_rate':[0.1, 0.01, 0.001],
        'n_estimators': [10, 50, 100, 500],
        'subsample':[1.0, 0.75, 0.5, 0.25],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'min_samples_split':[2, 5, 10, 20, None],
        'min_samples_leaf':[1, 5, 10, 20, None]
    },
    'XGBRegressor': {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'eta':[0.3, 0.03, 0.003],
        'gamma': [0, 2, 5, 10],
        'max_depth': [1, 3, 6, 10],
    },
    'CatBoostRegressor': {
        'verbose':[0],
        'iterations':[30, 50, 100, 150],
        'learning_rate':[0.3, 0.03, 0.003],
        'depth':[1, 3, 6, 10],
        'l2_leaf_reg':[0.1, 0.9, 1.0, 1.5],
    }
}