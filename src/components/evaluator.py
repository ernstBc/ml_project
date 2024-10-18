import os
import sys
import pandas as pd
from src.logger import logging
from src.handle_exception import CustomException
from src.config.config import BEST_MODEL_PATH, MODEL_HISTORY_PATH
from src.utils import save_artifact
from dataclasses import dataclass

from typing import Dict

@dataclass
class EvaluatorConfig:
    best_model_path: str=BEST_MODEL_PATH
    models_history:str = MODEL_HISTORY_PATH

class Evaluator:
    def __init__(self):
        self.evaluator_config=EvaluatorConfig()

    def read_history(self):
        try:
            if not os.path.exists(self.evaluator_config.models_history):
                return (None, None)
            
            hist=pd.read_csv(self.evaluator_config.models_history)
            best_score=hist.sort_values(by='mae', ascending=False)
            best_score=best_score.loc[0, 'mae']

            return hist, best_score
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def eval_model(self, model, metrics_report, model_name, best_score):
        try:
    
            hist, history_score=self.read_history()
    
            if hist is None:
                data=MetricsExample(metrics_report=metrics_report, model_name=model_name)
                logging.info('Creacion del registro de modelos.')
                data_formated=data.formating_metrics()
                data_formated.to_csv(self.evaluator_config.models_history, index=False)
                save_artifact(self.evaluator_config.best_model_path, model)
                print('Model Blessing')
                logging.info('Creacion del registro de modelos.')

            else:
                if best_score<history_score:
                    logging.info('Nuevo mejor modelo registrado')
                    data=MetricsExample(metrics_report=metrics_report,
                                                model_name=model_name)
                    data_formated=data.formating_metrics()
                    new_hist=pd.concat([hist, data_formated], axis=0)
                    logging.info('Registro de mejores modelos actualizado')
                    new_hist.to_csv(self.evaluator_config.models_history, index=False)
                    
                    if os.path.exists(self.evaluator_config.best_model_path):
                        os.remove(self.evaluator_config.best_model_path)
                    
                    save_artifact(self.evaluator_config.best_model_path, model)
                    logging.info('Model Blessing')
                    print('Model Blessing')
                else:
                    logging.info('No Model Blessing')

        except Exception as e:
            logging.info('Error al intentar evaluar el modelo con los registros.')
            raise CustomException(e, sys)



class MetricsExample:
    def __init__(self,model_name:str, metrics_report:Dict):
        self.model_name=model_name
        self.mae=metrics_report['test_metrics']['mae']
        self.mse=metrics_report['test_metrics']['mse']
        self.rmse=metrics_report['test_metrics']['rmse']
        self.r2_score=metrics_report['test_metrics']['r2_score']

        if "best_params" in metrics_report:
            self.params=metrics_report["best_params"]
        else:
            self.params='default'


    def formating_metrics(self):
        data={
            'model_name':[self.model_name],
            'mae':[self.mae],
            'mse':[self.mse],
            'rmse':[self.rmse],
            'r2_score':[self.r2_score],
            'params':[self.params]
        }
        df=pd.DataFrame(data)
        return df