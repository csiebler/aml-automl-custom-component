import os
import pytest

import automl_helper
from azureml._restclient.exceptions import ServiceException

def test_get_automl_run():
    experiment_name = "automl-classification"
    run_id = "AutoML_fea8ebb2-8480-4488-839f-49118530b230_2"
    ws = automl_helper.get_workspace()
    run = automl_helper.get_automl_run(ws, experiment_name, run_id)
    assert run.id == run_id
    
def test_get_automl_vision_run():
    experiment_name = 'automl-vision'
    run_id = 'AutoML_d92dae85-db77-4366-8363-a60e7093a38b_HD_1'
    ws = automl_helper.get_workspace()
    run = automl_helper.get_automl_run(ws, experiment_name, run_id)
    assert run.id == run_id
    
def test_get_automl_nonexistant_run():
    experiment_name = "automl-classification"
    run_id = "thisrundoesnotexist"
    ws = automl_helper.get_workspace()
    with pytest.raises(ServiceException):
        run = automl_helper.get_automl_run(ws, "automl-vision", run_id)

def test_get_automl_wrong_runtype():
    experiment_name = "automl-classification"
    run_id = "1eb03d84-2a19-4cda-b0a4-eaa3243c4305"
    ws = automl_helper.get_workspace()
    with pytest.raises(RuntimeError):
        run = automl_helper.get_automl_run(ws, "not-automl", run_id)