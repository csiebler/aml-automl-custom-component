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
        run_id="AutoML_556b350d-bf4e-42a2-9ea4-79e0dfbe4a84_HD_0")
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
        run_id="AutoML_d92dae85-db77-4366-8363-a60e7093a38b_HD_0")
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