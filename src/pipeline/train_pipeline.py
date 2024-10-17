import warnings
import argparse
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.components.trainer import Trainer

parser=argparse.ArgumentParser()
parser.add_argument('-t', '--tuning', action='store_true')


warnings.filterwarnings('ignore')

if __name__=='__main__':
    args=parser.parse_args()
    with_tuning=args.tuning

    ingestion=DataIngestion()
    transformer=DataTransformer()
    trainer=Trainer()

    train_path, test_path=ingestion.init_data_ingestion()
    train_arr, test_arr, preprocess_path=transformer.init_data_transformer(train_path, test_path)
    report,model,_=trainer.init_trainer(train_arr, test_arr, preprocess_path, with_tuning=with_tuning)
    model_name=model.__str__()[:-2]

    print(f'\nBest Model Metrics {model_name}:\n', report[model_name])