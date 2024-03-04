from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifacts

import os
import sys
import warnings
import pandas as pd
import numpy as np

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self):
        try:
            warnings.filterwarnings('ignore')
            df=pd.read_csv('data/adult/adult.data', header=None, sep=SEPRATOR)
            df.columns=COLUMNS
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR, index=False, header=True)

            data_ingestion_artifacts=DataIngestionArtifacts(
                raw_data_file_path=self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
            )

            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj=DataIngestion(DataIngestionConfig())
    obj.initiate_data_ingestion()