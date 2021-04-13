import os
import argparse
import pandas as pd
from sklearn.externals import joblib

import azureml.automl.core
from azureml.core import Workspace, Experiment, Run
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

import automl_helper

def parse_args():
    parser = argparse.ArgumentParser("AutoML-Scoring")
    parser.add_argument("--input_data", type=str, help="Input data")
    parser.add_argument("--predictions_data", type=str, help="Predictions data")
    parser.add_argument("--experiment", type=str, help="AutoML experiment name")
    parser.add_argument("--run_id", type=str, help="Run Id")
    return parser.parse_args()

def predict(args):

    # Load data that needs to be scored
    df = load_data_frame_from_directory(args.input_data).data

    # Connect to workspace
    ws = automl_helper.get_workspace()

    # Get AutoML run details
    automl_run = automl_helper.get_automl_run(ws, args.experiment, args.run_id)
    properties = automl_run.properties

    # Load AutoML model
    model = automl_helper.load_automl_model(automl_run)

    # Score data
    print("Using model to score input data...")
    
    isForecasting = ('fitted_pipeline' in automl_run.properties and automl_run.properties['fitted_pipeline'].startswith('ForecastingPipelineWrapper'))

    if (isForecasting):
        y_query = None
        if 'y_query' in df.columns:
            y_query = df.pop('y_query').values
        results = model.forecast(df, y_query)
        results = results[0]
    else:
        results = model.predict(df)
        
    results_df = pd.DataFrame(results, columns=['Predictions'])
    print(f"This is how your prediction data looks like:\n{results_df.head()}")

    # Write results back
    automl_helper.write_prediction_dataframe(args.predictions_data, results_df)

if __name__ == '__main__':
    args = parse_args()
    predict(args)