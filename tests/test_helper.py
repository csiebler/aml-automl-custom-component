import os
import pytest

import automl_helper
from azureml._restclient.exceptions import ServiceException

def test_get_automl_run():
    run_id = "AutoML_53a96f75-6f8b-48c5-8dc5-0944a3d5cf68_HD_3"
    ws = automl_helper.get_workspace()
    run = automl_helper.get_automl_run(ws, "automl-vision", run_id)
    assert run.id == run_id
    
def test_get_automl_nonexistant_run():
    run_id = "thisrundoesnotexist"
    ws = automl_helper.get_workspace()
    with pytest.raises(ServiceException):
        run = automl_helper.get_automl_run(ws, "automl-vision", run_id)