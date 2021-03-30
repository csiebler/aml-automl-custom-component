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
        experiment="automl-credit",
        run_id="AutoML_13528b38-bfe2-449e-9b43-967de35ecb3f_0")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files

def test_forecasting():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./tests/forecasting/input-sample",
        predictions_data=temp_dir.name,
        experiment="forecast_test",
        run_id="AutoML_4e2fe650-342f-456c-98ba-f9e6713cbf65_11")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files
    
def test_regression():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./tests/regression/input-sample",
        predictions_data=temp_dir.name,
        experiment="automl_regression",
        run_id="AutoML_3b9e2e2e-b2e4-4770-b261-0ec285f03276_2")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files
    
def test_fail_if_input_data_does_not_exist():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./thisdoesnotexist",
        predictions_data=temp_dir.name,
        experiment="automl-credit",
        run_id="AutoML_13528b38-bfe2-449e-9b43-967de35ecb3f_0")
    with pytest.raises(DirectoryLoadError):
        predict(args)
