import sys
from pathlib import Path
import pytest

cwd = Path.cwd()
main_dir = cwd.parent
sys.path.append(str(cwd))
sys.path.append(str(main_dir))

from get_model_prediction import *

@pytest.fixture()
def get_high_example():
	high = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 						 0.0, 0.0, 0.036, 0.0104, 0.048, 0.0, 0.0, 0.0, 0.0,
 						 0.0, 0.0, 0.038, 0.0, 0.034, 0.0323, 0.009, 0.0104]
	input = dict(zip(app_param_cols, high))
	yield input

@pytest.fixture()
def get_low_example():
	low = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
						 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012121212121212121, 0.0, 0.0,
						 0.0, 0.0, 0.01694915254237288, 0.022388059701492536, 0.004878048780487805,
						 0.0, 0.0, 0.0]
	low_input = dict(zip(app_param_cols, low))
	yield low_input

@pytest.fixture()
def get_very_high():
	vh = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 		  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1,
 		  0.0, 0.0, 2, 1, 0.0]
	vh_input = dict(zip(app_param_cols, vh))
	yield vh_input

class TestGetModelPrediction():
	def test_model_prediction(self, get_high_example, get_low_example, get_very_high):
		pred = get_model_prediction('./models/rf_model_d13.joblib', get_high_example)
		assert round(pred,3) == 0.563
		pred = get_model_prediction('./models/rf_model_d13.joblib', get_low_example)
		assert round(pred,3) == 0.331
		pred = get_model_prediction('./models/rf_model_d13.joblib', get_very_high)
		assert round(pred,3) == 0.611
		# testing with other model
		pred = get_model_prediction('./models/rf_model_d20.joblib', get_high_example)
		assert round(pred, 3) == 0.55
		pred = get_model_prediction('./models/rf_model_d20.joblib', get_low_example)
		assert round(pred,3) == 0.336
		pred = get_model_prediction('./models/rf_model_d20.joblib', get_very_high)
		assert round(pred,3) == 0.601
