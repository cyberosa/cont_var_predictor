# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
from data_processing import *
from get_model_prediction import *
from constants import app_param_cols
# creating a Flask app
app = Flask(__name__)


# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/', methods = ['GET', 'POST'])
def home():
	if(request.method == 'GET'):
		intro = "Welcome to the completion rate predictor. Please provide the following parameters"
		# name of the parameters defined in --> app_param_cols
		reqs = ''.join(app_param_cols)
		return jsonify({'intro': intro, 'reqs':reqs})


# A simple function to get the predictions of a model
# the needed variables are sent in the URL when we use GET
@app.route('/pred', methods = ['GET'])
def predict_conversion():
	try:
		pred_params = request.args.to_dict()
		if len(pred_params.keys()) < len(app_param_cols):
			msg = "Missing parameters in the request"
			return jsonify({'error_message': msg}), 400

		if pred_params is None:
			msg = "Wrong or missing parameters in the request"
			return jsonify({'error_message': msg}), 400

		for param in pred_params:
			pred_params[param] = float(pred_params[param])

		result = get_model_prediction('models/rfr_model.joblib',
								  pred_params)

		if result is None:
			msg = "There was an error when computing the prediction. Please check your parameters"
			return jsonify({'message': msg}), 400

		return jsonify({'completion_ rate': result})
	except Exception as e:
		msg = "There was an error when processing the request"
		return jsonify({'message': msg, 'error':e}), 400


# driver function
if __name__ == '__main__':

	app.run(debug = True)