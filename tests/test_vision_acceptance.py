import os
import pytest
import tempfile
from argparse import Namespace
from automl_vision_scoring import predict
from automl_vision_scoring import get_files_to_score
from azureml.studio.core.io.any_directory import DirectoryLoadError

def test_image_classification():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./tests/image_classification/",
        file_extension='jpg',
        include_subfolders='True',
        predictions_data=temp_dir.name,
        experiment="automl-vision",
        detailed_predictions='True',
        run_id="AutoML_53a96f75-6f8b-48c5-8dc5-0944a3d5cf68_HD_3")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files

def test_get_files_to_score():
    files = get_files_to_score(input_path='./tests/image_classification', file_extension='jpg', include_subfolders=True)
    assert len(files) == 3

    files = get_files_to_score(input_path='./tests/image_classification', file_extension='jpg', include_subfolders=False)
    assert len(files) == 2
    
    files = get_files_to_score(input_path='./tests/image_classification', file_extension='png', include_subfolders=True)
    assert len(files) == 0