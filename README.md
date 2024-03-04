## Implementation Steps
Create virtual environment
Install the requirements there 
Now Run `python src\pipeline\train_pipeline.py`
After Success full run, `logs` and `artifacts` folder will be create

 - artifacts 
    |
    | --DataIngestionArtifacts
    |           |
    |           | --data.csv
    | --DataPreprocessArtifacts
    |           |
    |           | --preprocess.csv
    | --DataTransformationArtifacts
    |           |
    |           | --final.csv
    |           | --test.csv
    |           | --train.csv
    | --ModelTrainerArtifacts
    |           |
    |           | --model.pkl
    | --ModelEvaluationArtifacts
    |           |
    |           | --metrics.json
    -------------------------------
- 


Prediction Pipeline will design Later 
