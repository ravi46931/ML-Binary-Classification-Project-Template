from src.constants import *
from src.entity.artifacts_entity import DataPreprocessArtifacts, DataTransformationArtifacts
from src.entity.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
from src.ml.standardization import DataStandardizer
from src.utils.utils import  frequency_encoding, label_encoding, data_split

import os
import sys
import pandas as pd
import numpy as np

class DataTransformation:
    def __init__(self, data_preprocess_artifact: DataPreprocessArtifacts, data_transformation_config: DataTransformationConfig):
        self.data_preprocess_artifact=data_preprocess_artifact
        self.data_transformation_config=data_transformation_config

    def data_transformation(self):
        try:
            df=pd.read_csv(self.data_preprocess_artifact.preprocess_data_file_path)

            dictionary=frequency_encoding(df, 'workclass')
            df['workclass'].replace(dictionary, inplace=True)

            dictionary=label_encoding(df, 'marital-status')
            df['marital-status'].replace(dictionary, inplace=True)

            dictionary=frequency_encoding(df, 'occupation')
            df['occupation'].replace(dictionary, inplace=True)

            dictionary=label_encoding(df, 'relationship')
            df['relationship'].replace(dictionary, inplace=True)

            dictionary=frequency_encoding(df, 'race')
            df['race'].replace(dictionary, inplace=True)

            df['sex'].replace(GENDER_MAP, inplace=True)

            dictionary=frequency_encoding(df, 'native-country')
            df['native-country'].replace(dictionary, inplace=True)

            df.rename(columns={'sex':'male'}, inplace=True)

            train_data, test_data=data_split(df)

            scaler=DataStandardizer()

            train_data[STANDARDIZATION_COLUMN] = scaler.fit_transform(train_data[STANDARDIZATION_COLUMN])

            test_data[STANDARDIZATION_COLUMN] = scaler.transform(test_data[STANDARDIZATION_COLUMN])

            return df, train_data, test_data

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self):
        try:
            df, transform_train_data, transform_test_data = self.data_transformation()
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)
            transform_train_data.to_csv(self.data_transformation_config.TRAIN_TRANSFORMED_FILE_PATH, index=False, header=True)
            transform_test_data.to_csv(self.data_transformation_config.TEST_TRANSFORMED_FILE_PATH, index=False, header=True)

            data_transformation_artifacts=DataTransformationArtifacts(
                train_transform_file_path=self.data_transformation_config.TRAIN_TRANSFORMED_FILE_PATH,
                test_transform_file_path=self.data_transformation_config.TEST_TRANSFORMED_FILE_PATH
            )

            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys)
