from src.exception import CustomException
from src.logger import logging
from src.constants import *
from src.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.utils import get_features_target
# from src.ml.model import LogisticRegression
from src.ml.metrics import accuracy, specifity, recall, f1_score, precision

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

class ModelEvaluation:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_artifacts:ModelTrainerArtifacts, model_evaluation_config: ModelEvaluationConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        self.model_evaluation_config = model_evaluation_config

    def modelevaluation(self):
        try:
            with open(self.model_trainer_artifacts.model_file_path, 'rb') as file:
                model = pickle.load(file)

            test_data = pd.read_csv(self.data_transformation_artifacts.test_transform_file_path)
            X_test, y_test = get_features_target(test_data, dependent_variable = DEPENDENT_VARIABLE)
            X_test.insert(0, 'x_0', np.ones(X_test.shape[0]))

            y_pred=model.predict(X_test.values)

            ACCURACY=accuracy(y_pred, y_test.values)
            SPECIFITY=specifity(y_pred, y_test.values)
            PRECISION=precision(y_pred, y_test.values)
            RECALL=recall(y_pred, y_test.values)
            F1_SCORE=f1_score(y_pred, y_test.values)

            metrics={
                'ACCURACY':ACCURACY,
                'SPECIFITY':SPECIFITY,
                'PRECISION': PRECISION,
                'RECALL': RECALL,
                'F1_SCORE':F1_SCORE
            }

            return metrics

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self):
        try:
            metrics = self.modelevaluation()
            os.makedirs(self.model_evaluation_config.MODEL_EVALUATION_ARTIFACTS_DIR, exist_ok=True)
            with open(self.model_evaluation_config.METRICS_FILE_PATH, 'w') as json_file:
                json.dump(metrics, json_file)

            model_evaluation_artifacts=ModelEvaluationArtifacts(self.model_evaluation_config.METRICS_FILE_PATH)
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)