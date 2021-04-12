import os, sys
import json
import argparse
import pandas as pd
from sklearn.externals import joblib
from operator import itemgetter

import azureml.automl.core
from azureml.core import Workspace, Experiment, Run
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

from azureml.contrib.automl.dnn.vision.common.model_export_utils import load_model, run_inference
from azureml.contrib.automl.dnn.vision.classification.inference.score import _score_with_model

# Parse args
def parse_args():
    parser = argparse.ArgumentParser("AutoML-Vision-Scoring")
    parser.add_argument("--input_data", type=str, help="Input data")
    parser.add_argument("--predictions_data", type=str, help="Predictions data")
    parser.add_argument("--experiment", type=str, help="AutoML experiment name")
    parser.add_argument("--detailed_predictions", type=str, help="Detailed predictions")
    parser.add_argument("--run_id", type=str, help="Run Id")
    return parser.parse_args()

def predict(args):

    # TODO: Add file extension flag and subdirectory mode
    # List files that need to be scored
    files = os.listdir(args.input_data)
    print(f"Files: {files}")

    # Get AutoML Vision run
    run = Run.get_context()
    if (isinstance(run, azureml.core.run._OfflineRun)):
        ws = Workspace.from_config()
    else:
        ws = run.experiment.workspace
    print(f"Retrieved access to workspace {ws}")

    # Try to read run details
    try:
        experiment = Experiment(ws, args.experiment)
        automl_run = Run(experiment, args.run_id)
        properties = automl_run.properties
    except Exception as e:
        raise

    # TODO: Check if run is actually an AutoML Vision run
    if (properties['runTemplate'] != "automl_child"):
        raise RuntimeError(f"Run with run_id={args.run_id} is a not an AutoML run!")

    # TODO: Support also other types
    TASK_TYPE = 'image-classification'
    print("Downloading AutoML Vision model...")
    automl_run.download_file('outputs/model.pt', output_file_path='./')
    model = load_model(TASK_TYPE, './model.pt')

    # Score data
    print("Using vision model to score input data...")
    
    results = {
        'filename': [],
        'prediction': [],
    }
    
    # TODO: add detailed predictions if args.detailed_predictions == true

    for file in files:
        print(f"Scoring file {file}")
        file_path = os.path.join(args.input_data, file)      
        data = open(file_path, 'rb').read()
        result = run_inference(model, data, _score_with_model)
        result = json.loads(result)
        print(result)
        
        index, element = max(enumerate(result['probs']), key=itemgetter(1))
        prediction_class = result['labels'][index]
        results['filename'].append(file)
        results['prediction'].append(prediction_class)
        
    results_df = pd.DataFrame(results)
        
    print("This is how your data looks like:")
    print(results_df.head())

    # Write results back
    print("Writing predictions back...")
    os.makedirs(args.predictions_data, exist_ok=True)
    save_data_frame_to_directory(args.predictions_data, results_df)

if __name__ == '__main__':
    args = parse_args()
    predict(args)