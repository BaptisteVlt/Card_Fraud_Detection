import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # In case there is an error when importing logger and exception below
from exception import CustomException
from logger import logging
from components.data_transformation import DataTransformation, DataTransformationConfig
from components.model_trainer import ModelTrainerConfig, ModelTrainer
from components.data_ingestion import DataIngestion, DataIngestionConfig

class TrainPipeline:
    def __init__(self):
        pass


    def train(self):
        try:
            logging.info("Start training pipeline")
            
            #Data Ingestion
            try:
                obj=DataIngestion()
                train_data, test_data=obj.initiate_data_ingestion()
                logging.info("Data ingestion completed")
            except Exception as e:
                raise CustomException(e, sys)



            #Data Transformation
            try:
                data_transformation=DataTransformation()
                train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
                logging.info("Data Transformation completed")
            except Exception as e:
                raise CustomException(e, sys)

            #Training Model
            try:
                model_trainer=ModelTrainer()
                best_model_score=model_trainer.initiate_model_trainer(train_arr, test_arr )
                logging.info(f"Model training completed. Best Model Score:{best_model_score}")
            except Exception as e:
                raise CustomException(e, sys)

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.train()
    except Exception as e:
        raise CustomException(e, sys)