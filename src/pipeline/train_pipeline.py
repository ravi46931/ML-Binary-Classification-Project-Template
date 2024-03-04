from src.components.data_ingestion import DataIngestion
from src.components.data_preprocess import DataPreprocess
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import DataIngestionConfig, DataPreprocessConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig



if __name__=="__main__":
    obj1=DataIngestion(DataIngestionConfig())
    data_ingestion_artifact=obj1.initiate_data_ingestion()

    obj2=DataPreprocess(data_ingestion_artifact, DataPreprocessConfig())
    data_preprocess_artifact=obj2.initiate_data_preprocessing()

    obj3=DataTransformation(data_preprocess_artifact, DataTransformationConfig())
    data_transformation_artifacts=obj3.initiate_data_transformation()

    obj4=ModelTrainer(data_transformation_artifacts, ModelTrainerConfig())
    model_trainer_artifact=obj4.initiate_model_training()

    obj5=ModelEvaluation(data_transformation_artifacts, model_trainer_artifact, ModelEvaluationConfig())
    model_evaluation_artifacts=obj5.initiate_model_evaluation()
    # print(data_ingestion_artifact)