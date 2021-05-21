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
 						 0.0, 0.0, 0.038, 0.0, 0.034, 0.0323, 0.009, 0.0104,
 						 0.0, 0.0, 0.0, 0.0074, 0.031]
	input = dict(zip(app_param_cols, high))
	yield input

@pytest.fixture()
def get_low_example():
	low = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
						 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012121212121212121, 0.0, 0.0,
						 0.0, 0.0, 0.01694915254237288, 0.022388059701492536, 0.004878048780487805,
						 0.0, 0.0, 0.0, 0.034571, 0.0, 0.099990, 0.0, 0.0]
	low_input = dict(zip(app_param_cols, low))
	yield low_input

@pytest.fixture()
def get_very_high():
	vh = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
 		  0.8064516129032258, 0.0, 0.6060606060606061, 0.0, 0.04, 0.09, 0.0, 0.0847457627118644,
 		  0.0, 0.0975609756097561, 0.0, 0.890, 0.0, 0.056, 0.32310, 0.08, 0.0, 0.17467248908296944]
	vh_input = dict(zip(app_param_cols, vh))
	yield vh_input

class TestGetModelPrediction():
	def test_model_prediction(self, get_high_example, get_low_example, get_very_high):
		pred = get_model_prediction('./models/rfr_model.joblib', get_high_example)
		assert round(pred,3) == 0.533
		pred = get_model_prediction('./models/rfr_model.joblib', get_low_example)
		assert round(pred,3) == 0.416
		pred = get_model_prediction('./models/rfr_model.joblib', get_very_high)
		assert round(pred,3) == 0.569
