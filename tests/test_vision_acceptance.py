import os
import pytest
import tempfile
from argparse import Namespace
from automl_vision_scoring import predict
from azureml.studio.core.io.any_directory import DirectoryLoadError

def test_image_classification():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./tests/image_classification/",
        predictions_data=temp_dir.name,
        experiment="automl-vision",
        run_id="AutoML_53a96f75-6f8b-48c5-8dc5-0944a3d5cf68_HD_3")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files
