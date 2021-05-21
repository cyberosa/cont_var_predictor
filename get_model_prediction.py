import joblib
from data_processing import *
import pandas as pd


def prepare_features(params_dict):
	'''
	It prepares the input parameters for the model, transforming and scaling
	the needed features.
	:param params_dict: dictionary with all needed parameters for the model
	:return: an array with all the model parameters in the right order
	'''
	# split first into two groups of features
	syn_f, scaled_f = group_features(params_dict)
	print("Preparing synthetic features")
	syn_f = transform_syn_features(syn_f)
	#print("Synthetic features {}".format(syn_f))
	print("Scaling numerical features")
	scaled_f = scale_num_features(scaled_f)
	#print("Scaled features {}".format(scaled_f))
	array_values = list(syn_f.values())
	array_values.extend(list(scaled_f.values()))
	return array_values

def get_model_prediction(model_filename, params_dict):
	'''
	For a given model it gives the prediction for the given parameters
	:param model_filename: filename path
	:param params_dict: dictionary with all the needed parameters for the model
	:return: the prediction given by the model
	'''
	# Prepare the parameters before passing them to the model
	try:
		print("Preparing the features for the model")
		x_params = prepare_features(params_dict)

		# load the model
		print("Model filename {}".format(model_filename))
		model = joblib.load(model_filename)

		# preparing the input dataset for the model
		print("Preparing X_test for the model")
		X_test = pd.DataFrame(columns=input_model_cols)
		if len(x_params) != len(input_model_cols):
			print("Wrong length of the input parameters for the model")
			return None
		X_test.loc[0, :] = x_params
		print("X_test")
		print(X_test)
		result = model.predict(X_test)
		print("Model returned a prediction {}".format(result[0]))
		return result[0]
	except Exception as e:
		print(e)
		print("Error when trying to generate the prediction")
		return None
