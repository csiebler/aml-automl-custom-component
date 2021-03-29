import os
import argparse
import pandas as pd
from sklearn.externals import joblib

import azureml.automl.core
from azureml.core import Workspace, Experiment, Run
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

# Parse args
parser = argparse.ArgumentParser("AutoML-Scoring")
parser.add_argument("--input_data", type=str, help="Input data")
parser.add_argument("--predictions_data", type=str, help="Predictions data")
parser.add_argument("--experiment", type=str, help="AutoML experiment name")
parser.add_argument("--run_id", type=str, help="Run Id")
args = parser.parse_args()

# Load data that needs to be scored
df = load_data_frame_from_directory(args.input_data).data

# Get AutoML run
run = Run.get_context()
if (isinstance(run, azureml.core.run._OfflineRun)):
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace
    
print(f"Retrieved access to workspace {ws}")

try:
    experiment = Experiment(ws, args.experiment)
    automl_run = Run(experiment, args.run_id)
    properties = automl_run.properties
except Exception as e:
    raise

if (properties['runTemplate'] != "automl_child"):
    raise RuntimeError(f"Run with run_id={args.run_id} is a not an AutoML run!")

isForecasting = ('fitted_pipeline' in automl_run.properties and automl_run.properties['fitted_pipeline'].startswith('ForecastingPipelineWrapper'))

print("Downloading AutoML model...")
automl_run.download_file('outputs/model.pkl', output_file_path='./')
model_path = './model.pkl'
model = joblib.load(model_path)

# Score data
print("Using model to score input data...")

if (isForecasting):
    y_query = None
    if 'y_query' in data.columns:
        y_query = data.pop('y_query').values
    results = model.forecast(df, y_query)
else:
    results = model.predict(df)

results_df = pd.DataFrame(results, columns=['Predictions'])

print("This is how your data looks like:")
print(results_df.head())

# Write results back
print("Writing predictions back...")
os.makedirs(args.predictions_data, exist_ok=True)
save_data_frame_to_directory(args.predictions_data, results_df)
