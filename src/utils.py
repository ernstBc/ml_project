import pickle
import os
import sys
import json
from typing import Dict
from src.logger import logging
from src.handle_exception import CustomException
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             root_mean_squared_error,
                             r2_score)

def save_artifact(artifact_path, object, is_json=False):
    try: 
        dir_path=os.path.dirname(artifact_path)
        os.makedirs(dir_path, exist_ok=True)

        if is_json:
            with open(artifact_path, "w") as file: 
                json.dump(object, file, sort_keys=True, indent=4)
        else:
            with open(artifact_path, 'wb') as file:
              pickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info('Artefacto guardado en artifacts')
    except Exception as e:
        logging.info('Error al crear el artifacto')
        raise CustomException(e, sys)


def load_artifact(artifact_path):
    try:
        with open(artifact_path,'rb') as handle:
            artifact=pickle.load(handle)
        
        logging.info(f'Artefacto cargado correctamente: {artifact_path}')
        return artifact
    
    except Exception as e:
        logging.info(f'Error cargando artefacto: {artifact_path}')


def eval_reg_model(y, preds, split:str=None):
    mse=mean_squared_error(y, preds)
    rmse=root_mean_squared_error(y, preds)
    mae=mean_absolute_error(y, preds)
    r2=r2_score(y,preds)

    if split is not None:
        metrics={
            f"mse_{split}":mse,
            f"rmse_{split}":rmse,
            f"mae_{split}":mae,
            f"r2_{split}":r2

        }
        return metrics
    
    return {'mse':mse,
            'mae':mae,
            'rmse':rmse,
            'r2_score':r2}


def fit_models(x_train, x_test, y_train,y_test, models:Dict):
    try:
        models_metrics={}

        best_model=''
        best_mae_score=None
        for model_name, model in models.items():
            model.fit(x_train,y_train)
            train_preds=model.predict(x_train)
            test_preds=model.predict(x_test)

            train_metrics=eval_reg_model(y_train, train_preds)
            test_metrics=eval_reg_model(y_test, test_preds, )

            model_report={'train_metrics': train_metrics,
                          'test_metrics': test_metrics}
            
            models_metrics[model_name]=model_report

            if best_mae_score==None:
                best_mae_score=test_metrics['mae']
                best_model=model_name
            else:
                if test_metrics['mae']<best_mae_score:
                    best_mae_score=test_metrics['mae']
                    best_model=model_name
        
        return models_metrics, best_mae_score, best_model

    except Exception as e:
        raise CustomException(e, sys)

    