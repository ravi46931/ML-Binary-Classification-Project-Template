##############################################
# common constants
ARTIFACTS='artifacts'

# Data Ingestion constant
RAW_FILE_NAME= 'data.csv'
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
SEPRATOR = ', '
COLUMNS=[
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income'
]

# Data Preprocessing constant
DATA_PREPROCESS_ARTIFACTS_DIR = "DataPreprocessArtifacts"
PREPROCESS_FILE_NAME= 'preprocess.csv'
VALUE_REPLACE='?'
WORKCLASS_DROP_VALUES = ['Without-pay','Never-worked']
MARITAL_STATUS_DROP_VALUES = ['Married-AF-spouse']
OCCUPATION_DROP_VALUES = ['Armed-Forces']
INCOME_MAP = {'>50K':1,'<=50K': 0}


# Data Transformation Config
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRANSFORMED_FILE_NAME='final.csv'
TRAIN_TRANSFORMED_FILE_NAME="train.csv"
TEST_TRANSFORMED_FILE_NAME="test.csv"
GENDER_MAP = {'Female':0, 'Male':1}
STANDARDIZATION_COLUMN=[
    'age',
    'fnlwgt', 
    'education-num', 
    'marital-status', 
    'relationship', 
    'capital-gain', 
    'capital-loss', 
    'hours-per-week'
]

# Model Trainer Config
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
MODEL_FILE_NAME='model.pkl'
DEPENDENT_VARIABLE = 'income'
ITERATIONS = 15
ALPHA = 0.25
THRESOLD_VALUE = 0.5

# Model Evaluation Constants
MODEL_EVALUATION_ARTIFACTS_DIR="ModelEvaluationArtifacts"
METRICS_FILE_NAME='metrics.json'
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0



