import os
import pytest
import tempfile
from argparse import Namespace
from automl_scoring import predict
from azureml.studio.core.io.any_directory import DirectoryLoadError

def test_classification():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./tests/classification/input-sample",
        predictions_data=temp_dir.name,
        experiment="automl-classification",
        run_id="AutoML_fea8ebb2-8480-4488-839f-49118530b230_2")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files

def test_forecasting():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./tests/forecasting/input-sample",
        predictions_data=temp_dir.name,
        experiment="automl-forecast",
        run_id="AutoML_1ec30054-0fac-4b9d-9624-91e827e30ed1_0")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files
    
def test_regression():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./tests/regression/input-sample",
        predictions_data=temp_dir.name,
        experiment="automl-regression",
        run_id="AutoML_03af1f3d-6323-4510-a8de-f3970bb0804a_0")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files
    
def test_fail_if_input_data_does_not_exist():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./thisdoesnotexist",
        predictions_data=temp_dir.name,
        experiment="automl-classification",
        run_id="AutoML_fea8ebb2-8480-4488-839f-49118530b230_2")
    with pytest.raises(DirectoryLoadError):
        predict(args)
