import os
import sys
from pathlib import Path


list_of_files=[
    'src/components/__init__.py',
    'src/components/data_ingestion.py',
    'src/components/data_transformation.py',
    'src/components/model_trainer.py',
    'src/components/model_evaluation.py',
    'src/entity/__init__.py',
    'src/entity/config_entity.py',
    'src/entity/artifacts_entity.py',
    'src/constants/__init__.py',
    'src/logger/__init__.py',
    'src/exception/__init__.py',
    'src/utils/__init__.py',
    'src/utils/utils.py',
    'src/pipeline/train_pipeline.py',
    'src/pipeline/prediction_pipeline.py',
    'src/pipeline/__init__.py',
    "src/ml/__init__.py",
    "src/ml/model.py",
    "src/ml/metrics.py",
    "src/ml/stndardization.py",
    'requirements.txt',
    'README.md',
    'setup.py',
    'app.py',
    'demo.py'
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir, filename=os.path.split(filepath)
    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0): # This line will be responsible for ERROR
        with open(filepath, "w") as f:
            pass
    else:
        pass