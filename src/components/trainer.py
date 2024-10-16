import os
import datetime
import sys

from src.logger import logging
from src.handle_exception import CustomException
from src.utils import save_artifact, load_artifact, fit_models
from dataclasses import dataclass

# modelos
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class TrainerConfig:
    trainer_artifact_path:str=os.path.join('artifacts',
                                           'models',
                                           'models',
                                           f'model-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl')
    trainer_metrics_path:str=os.path.join('artifacts',
                                           'models',
                                           'metrics',
                                           f'models-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json')

class Trainer:
    def __init__(self):
        self.trainer_config=TrainerConfig()

    def init_trainer(self, train_data, test_data, preprocessor_path):
        try:
            logging.info('Init trainer')
            preprocess=load_artifact(preprocessor_path)

            logging.info('Separar train/test set')
            x_train, x_test=train_data[:, :-1], test_data[:, :-1]
            y_train, y_test=train_data[:, -1], test_data[:, -1]


            models={
                'LinearRegression':LinearRegression(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0)
                }
            
            report, best_mae_score, best_model_name=fit_models(x_train=x_train,
                                                          x_test=x_test,
                                                          y_train=y_train,
                                                          y_test=y_test,
                                                          models=models)
            best_model=models[best_model_name].fit(x_train, y_train)

            logging.info('Reporte creado')
            save_artifact(self.trainer_config.trainer_metrics_path, report, is_json=True)
            
            # guardor el modelo
            try:
                if best_mae_score<12.0:
                    save_artifact(self.trainer_config.trainer_artifact_path, best_model)
                else:
                    1/0
            except Exception as e:
                raise CustomException('Ningun modelo supero el minimo requerido', sys)
            
            
            logging.info('Modelo guardado como artefacto')
            return report, best_model, best_mae_score

        except Exception as e:
            logging.info('Error al entrenar los modelos')
            raise CustomException(e, sys)
