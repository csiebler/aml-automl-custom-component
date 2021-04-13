import os, sys
import glob
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
    parser.add_argument("--include_subfolders", type=str, help="List files in subfolders")
    parser.add_argument("--file_extension", type=str, help="File extension, e.g. jpg or png")
    parser.add_argument("--predictions_data", type=str, help="Predictions data")
    parser.add_argument("--experiment", type=str, help="AutoML experiment name")
    parser.add_argument("--detailed_predictions", type=str, help="Detailed predictions")
    parser.add_argument("--run_id", type=str, help="Run Id")
    return parser.parse_args()

def get_files_to_score(input_path, file_extension='jpg', include_subfolders=True):
    if include_subfolders:
        search_path = os.path.join(input_path, '**/*.' + file_extension)
        print(search_path)
        files = glob.glob(search_path, recursive=True)
    else:
        search_path = os.path.join(input_path, '*.' + file_extension)
        print(search_path)
        files = glob.glob(search_path)
    return files

def predict(args):

    # List files that need to be scored
    include_subfolders = (args.include_subfolders.lower() in ('yes', 'true', 't', 'y', '1'))
    file_extension = args.file_extension.lower()
    files = get_files_to_score(args.input_data, file_extension, include_subfolders)
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
    
    for file in files:
        print(f"Scoring file {file}")
        #file_path = os.path.join(args.input_data, file)      
        data = open(file, 'rb').read()
        prediction_result = run_inference(model, data, _score_with_model)
        prediction_result = json.loads(prediction_result)
        print(prediction_result)
        
        # Get argmax of prediction
        index, element = max(enumerate(prediction_result['probs']), key=itemgetter(1))
        prediction_class = prediction_result['labels'][index]
        
        # Add results to output results
        results['filename'].append(file)
        results['prediction'].append(prediction_class)
        
        # Add details if needed
        if (args.detailed_predictions.lower() in ('yes', 'true', 't', 'y', '1')):
            for i, label in enumerate(prediction_result['labels']):
                if label not in results:
                    results[label] = []
                results[label].append(prediction_result['probs'][i])
        
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