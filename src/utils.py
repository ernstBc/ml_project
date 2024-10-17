import pickle
import os
import sys
import json
from typing import Dict
from src.logger import logging
from src.handle_exception import CustomException
from sklearn.model_selection import GridSearchCV
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


def fit_models(x_train, x_test, y_train,y_test, models:Dict, params:Dict=None):
    try:
        models_metrics={}

        best_model=None
        best_mae_score=None
        for model_name, model in models.items():
            if params is None:
                model.fit(x_train,y_train)
                
            else:
                model_params=params[model_name]

                grid=GridSearchCV(model, model_params, scoring='neg_mean_squared_error',cv=3, n_jobs=-1)
                grid.fit(x_train, y_train)
                
                model= grid.best_estimator_

            train_preds=model.predict(x_train)
            test_preds=model.predict(x_test)

            train_metrics=eval_reg_model(y_train, train_preds)
            test_metrics=eval_reg_model(y_test, test_preds, )

            model_report={'train_metrics': train_metrics,
                          'test_metrics': test_metrics}
            
            if params is not None:
                model_report['best_params']=grid.best_params_
            
            models_metrics[model_name]=model_report

            if best_mae_score==None:
                best_mae_score=test_metrics['mae']
                best_model=model
        
            else:
                if test_metrics['mae']<best_mae_score:
                    best_mae_score=test_metrics['mae']
                    best_model=model
        
        return models_metrics, best_mae_score, best_model

    except Exception as e:
        raise CustomException(e, sys)
    

def get_lasted_model(models_path:str) -> str:
    path_models=os.listdir(models_path)
    lasted_model_name=sorted(path_models)[-1]
    lasted_model_dir=os.path.join(models_path, lasted_model_name)
    return load_artifact(lasted_model_dir)


if __name__=='__main__':
    print(get_lasted_model(r'artifacts\models\models'))