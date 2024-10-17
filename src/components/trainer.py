import os
import datetime
import sys

from src.logger import logging
from src.handle_exception import CustomException
from src.utils import save_artifact, load_artifact, fit_models
from src.config.config import MODEL_FOLDER_PATH, METRICS_FOLDER_PATH
from src.models_utils import models, models_params
from dataclasses import dataclass


@dataclass
class TrainerConfig:
    trainer_artifact_path:str=os.path.join(MODEL_FOLDER_PATH,
                                           f'model-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl')
    trainer_metrics_path:str=os.path.join(METRICS_FOLDER_PATH,
                                           f'models-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json')


class Trainer:
    def __init__(self):
        self.trainer_config=TrainerConfig()


    def init_trainer(self, train_data, test_data, preprocessor_path, with_tuning=False):
        try:
            logging.info('Init trainer')
            preprocess=load_artifact(preprocessor_path)

            logging.info('Separar train/test set')
            x_train, x_test=train_data[:, :-1], test_data[:, :-1]
            y_train, y_test=train_data[:, -1], test_data[:, -1]


            if with_tuning:
                logging.info('INIT training sin tuning')
                report, best_mae_score, best_model=fit_models(x_train=x_train,
                                                          x_test=x_test,
                                                          y_train=y_train,
                                                          y_test=y_test,
                                                          models=models,
                                                          params=models_params)
            else:
                logging.info('INIT training con tuning')
                report, best_mae_score, best_model=fit_models(x_train=x_train,
                                                          x_test=x_test,
                                                          y_train=y_train,
                                                          y_test=y_test,
                                                          models=models)

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
