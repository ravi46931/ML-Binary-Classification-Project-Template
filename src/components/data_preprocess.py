from src.constants import *
from src.entity.artifacts_entity import DataPreprocessArtifacts, DataIngestionArtifacts
from src.entity.config_entity import DataPreprocessConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.utils import modify_null_values

import os
import sys
import pandas as pd
import numpy as np

class DataPreprocess:
    def __init__ (self, data_ingestion_artifact: DataIngestionArtifacts, data_preprocess_config: DataPreprocessConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_preprocess_config = data_preprocess_config

    def data_preprocess(self):
        try:
            df=pd.read_csv(self.data_ingestion_artifact.raw_data_file_path)

            # Replacing with the null values
            df.replace(VALUE_REPLACE, np.nan, inplace=True)

            # Drop some values from the columns
            df= df[~(df['workclass'].isin(WORKCLASS_DROP_VALUES))]

            df=modify_null_values(df, 'workclass')

            # Drop education column
            df.drop(['education'], axis=1, inplace=True)

            df= df[~(df['marital-status'].isin(MARITAL_STATUS_DROP_VALUES))]

            df= df[~(df['occupation'].isin(OCCUPATION_DROP_VALUES))]

            df=modify_null_values(df, 'occupation')

            df=modify_null_values(df, 'native-country')

            df.dropna(subset=['native-country'], inplace=True)

            df['income'].replace(INCOME_MAP, inplace=True)

            return df 
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_preprocessing(self):
        try:
            df=self.data_preprocess()
            os.makedirs(self.data_preprocess_config.DATA_PREPROCESS_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_preprocess_config.PREPROCESSED_DATA_ARTIFACTS_DIR, index=False, header=True)

            data_preprocess_artifact=DataPreprocessArtifacts(
                preprocess_data_file_path=self.data_preprocess_config.PREPROCESSED_DATA_ARTIFACTS_DIR
            )

            return data_preprocess_artifact
        
        except Exception as e:
            raise CustomException(e, sys)