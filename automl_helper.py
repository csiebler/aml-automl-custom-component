import os
import json
import azureml.automl.core

from sklearn.externals import joblib

from azureml.core import Workspace, Experiment, Run
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

def get_workspace():
    run = Run.get_context()
    if (isinstance(run, azureml.core.run._OfflineRun)):
        ws = Workspace.from_config()
    else:
        ws = run.experiment.workspace
    print(f"Retrieved access to workspace {ws}")
    return ws

def get_automl_run(workspace, experiment, run_id):
    try:
        experiment = Experiment(workspace, experiment)
        automl_run = Run(experiment, run_id)

        if ('runTemplate' not in automl_run.properties or automl_run.properties['runTemplate'] != "automl_child"):
            raise RuntimeError(f"Run with run_id={run_id} is a not an AutoML run!")

        # Get parent run
        parent_run = automl_run.parent
        while (parent_run.parent is not None):
            parent_run = parent_run.parent
        
        if (parent_run.type != 'automl'):
            raise RuntimeError(f"Only AutoML runs are supported, this run is of type {parent_run.type}!")
    except Exception as e:
        raise

    return automl_run

def load_automl_model(automl_run):
    print("Downloading AutoML model...")
    automl_run.download_file('outputs/model.pkl', output_file_path='./')
    model_path = './model.pkl'
    model = joblib.load(model_path)
    return model
    
def load_automl_vision_model(automl_run, task):
    
    from azureml.contrib.automl.dnn.vision.common.model_export_utils import load_model, run_inference
    
    if (task == 'image-classification'):
        from azureml.contrib.automl.dnn.vision.classification.inference.score import _score_with_model
        model_settings = {}
    elif (task == 'image-object-detection'):
        from azureml.contrib.automl.dnn.vision.object_detection_yolo.writers.score import _score_with_model
        model_settings = {"box_score_thresh": 0.4, "box_iou_thresh": 0.5}
    else:
        raise RuntimeError("Only task types image-classification and image-object-detection are currently supported!")

    print("Downloading AutoML Vision model...")
    automl_run.download_file('outputs/model.pt', output_file_path='./')
    print("Loding AutoML Vision model...")
    model = {
        'model': load_model(task, './model.pt', **model_settings),
        'scorer': _score_with_model
    }
    return model

def score_automl_vision_model(model, file_path):
    print(f"Scoring file {file_path}")
    data = open(file_path, 'rb').read()
    prediction_result = run_inference(model['model'], data, model['scorer'])
    prediction_result = json.loads(prediction_result)
    print(f"Predicted: {prediction_result} for {file_path}")
    return prediction_result

def write_prediction_dataframe(dir_path, dataframe):
    print("Writing predictions back...")
    os.makedirs(dir_path, exist_ok=True)
    save_data_frame_to_directory(dir_path, dataframe)