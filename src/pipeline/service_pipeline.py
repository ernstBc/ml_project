import pandas as pd
import sys

from src.handle_exception import CustomException
from src.logger import logging
from src.utils import load_artifact
from src.config.config import PREPROCESS_FILE_PATH, BEST_MODEL_PATH


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocess=load_artifact(PREPROCESS_FILE_PATH)
            model=load_artifact(BEST_MODEL_PATH)

            data_process=preprocess.transform(features)

            preds=model.predict(data_process)

            return preds
        except Exception as e:
            raise CustomException('Error al tratar de realizar la predicccion de datos. Verifique que hay al menos un modelo entrenado.', sys)



class ExampleData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_education:str,
                 lunch:str,
                 test_preparation_course:str):
        
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_education=parental_level_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course

    
    def format_data_as_frame(self):
        try:
            data_dict={
                'gender':[self.gender],
                'race/ethnicity':[self.race_ethnicity],
                'parental level of education':[self.parental_level_education],
                'lunch':[self.lunch],
                'test preparation course':[self.test_preparation_course]
            }
            
            return pd.DataFrame(data_dict)
        except Exception as e:
            logging.info('Error en service_pipeline.py. Imposible transformar los datos en un dataframe')