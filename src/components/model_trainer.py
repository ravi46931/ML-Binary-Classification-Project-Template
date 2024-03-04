from src.exception import CustomException
from src.logger import logging
from src.constants import *
from src.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts
from src.entity.config_entity import ModelTrainerConfig
from src.utils.utils import get_features_target
from src.ml.model import LogisticRegression

import os
import sys
import pickle
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def training(self):
        try:
            train_data = pd.read_csv(self.data_transformation_artifacts.train_transform_file_path)
            

            # income
            X_train, y_train = get_features_target(train_data, dependent_variable = DEPENDENT_VARIABLE)
            

            # Insert first column to 1's as we don't explicitly calculate the bias
            X_train.insert(0, 'x_0', np.ones(X_train.shape[0]))
            # 

            model=LogisticRegression()
            model.fit(X_train.values,y_train.values)

            return model

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_training(self):
        try:
            model=self.training()
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)

            with open(self.model_trainer_config.MODEL_FILE_PATH, 'wb') as file:
                pickle.dump(model, file)

            model_trainer_artifact=ModelTrainerArtifacts(
                model_file_path=self.model_trainer_config.MODEL_FILE_PATH
            )

            return model_trainer_artifact
                    
        except Exception as e:
            raise CustomException(e, sys)