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
        task_type='image-classification',
        predictions_data=temp_dir.name,
        experiment="automl-vision",
        run_id="AutoML_53a96f75-6f8b-48c5-8dc5-0944a3d5cf68_HD_3")
    predict(args)
    files = os.listdir(temp_dir.name)
    assert '_data.parquet' in files

def test_object_detection():
    temp_dir = tempfile.TemporaryDirectory()
    args = Namespace(input_data="./tests/object_detection/",
        file_extension='jpg',
        include_subfolders='True',
        task_type='image-object-detection',
        predictions_data=temp_dir.name,
        experiment="automl-vision",
        run_id="AutoML_9f62d9c1-acf8-4829-aaba-a0b4ca32ce7a_HD_0")
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