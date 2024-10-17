import os
import pandas as pd

from src.logger import logging
from src.utils import load_artifact, get_lasted_model
from src.config.config import PREPROCESS_FILE_PATH, MODEL_FOLDER_PATH


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        preprocess=load_artifact(PREPROCESS_FILE_PATH)
        model=get_lasted_model(MODEL_FOLDER_PATH)

        data_process=preprocess.transform(features)

        preds=model.predict(data_process)

        return preds



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