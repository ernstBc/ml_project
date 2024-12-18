import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.handle_exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    
    def init_data_ingestion(self):
        logging.info("Iniciando Data Ingestion")
        try:
            df=pd.read_csv(os.path.join('notebooks','data','StudentsPerformance.csv'))
            logging.info('Datos cargados como DataFrame')

            # crea dir
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            #dividir los datos en subsets
            train_data, test_data=train_test_split(df, test_size=0.2, random_state=86)

            # guardar csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Completado Data ingestion')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        
        except Exception as e:
            logging.info('Error al cargar la base de datos.')
            raise CustomException(e, sys)