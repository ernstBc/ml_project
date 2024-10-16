import sys
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.logger import logging
from src.handle_exception import CustomException
from src.utils import save_artifact
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


@dataclass
class DataTransformConfig:
    preprocess_artifact_path:str =os.path.join('artifacts','preprocessor.pkl')


class DataTransformer:
    def __init__(self):
        self.data_transformation_config=DataTransformConfig()
    
    def create_transformer_artifact(self):
        try:
            cat_vars=[
                'gender',
                "race/ethnicity",
                "parental level of education",
                'lunch',
                "test preparation course"
            ]

            pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('encoder', OneHotEncoder(sparse_output=False))
                ]
            )
            logging.info('Pipeline creado.')
            logging.info(f'Variables utilizadas: {cat_vars}')
            preprocess=ColumnTransformer(
                [
                    ('Pipeline', pipeline, cat_vars)
                ]
            )

            return preprocess
        except Exception as e:
            logging.info('Fallo al crear el artifacto "transformer/preprocessor')
            raise CustomException(e, sys)
        
    def init_data_transformer(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Datos cargados para crear el preprocesador')
            preprocess=self.create_transformer_artifact()


            target_col='avg_score'
            ignore_vars=['math score', 'reading score','writing score']
            
            train_df[target_col]=(train_df[ignore_vars[0]]+train_df[ignore_vars[1]]+train_df[ignore_vars[2]]) / 3
            test_df[target_col]=(test_df[ignore_vars[0]]+test_df[ignore_vars[1]]+test_df[ignore_vars[2]]) / 3

            target_train_feature=train_df[target_col]
            target_test_feature=test_df[target_col]

            train_inputs=train_df.drop(ignore_vars + [target_col], axis=1)
            test_inputs=test_df.drop(ignore_vars + [target_col], axis=1)
            logging.info('Aplicar preprocesamiento a los datos de entrenamiento y test')

            train_data=preprocess.fit_transform(train_inputs)
            test_data=preprocess.transform(test_inputs)

            train_array=np.c_[train_data, np.array(target_train_feature)]
            test_array=np.c_[test_data, np.array(target_test_feature)]

            logging.info('Preprocess artifact creado')
            save_artifact(self.data_transformation_config.preprocess_artifact_path, preprocess)

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocess_artifact_path
            )
        except Exception as e:
            logging.info('Error: No fue posible crear el artefacto preprocessor')