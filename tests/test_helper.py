import os
import pytest

import automl_helper
from azureml._restclient.exceptions import ServiceException

experiment_name = "automl-classification"
run_id = "AutoML_fea8ebb2-8480-4488-839f-49118530b230_2"
run_id_nonexist = "thisrundoesnotexist"

def test_get_automl_run():
    ws = automl_helper.get_workspace()
    run = automl_helper.get_automl_run(ws, experiment_name, run_id)
    assert run.id == run_id
    
def test_get_automl_nonexistant_run():
    ws = automl_helper.get_workspace()
    with pytest.raises(ServiceException):
        run = automl_helper.get_automl_run(ws, "automl-vision", run_id_nonexist)