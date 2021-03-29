import os
import pytest
import tempfile
from argparse import Namespace

from automl_scoring import predict

def test_classification():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data= "./tests/classification/input-sample",
        predictions_data= temp_dir.name,
        experiment= "automl-credit",
        run_id= "AutoML_13528b38-bfe2-449e-9b43-967de35ecb3f_0")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files

def test_forecasting():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data= "./tests/forecasting/input-sample",
        predictions_data= temp_dir.name,
        experiment= "forecast_test",
        run_id= "AutoML_4e2fe650-342f-456c-98ba-f9e6713cbf65_11")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files