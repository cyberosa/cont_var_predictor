'''
Tools to clean, to transform and to manage data from a dataset
'''
import pandas as pd
import numpy as np
from constants import *
from sklearn.preprocessing import MinMaxScaler

def generate_training_df(input_data, output_name):
	'''
	Method to generate a new training dataset and save it as a pickle file
	:param input_data: full path to the input data
	:param output_name: output filename
	:return:
	'''

	df = pd.read_csv(input_data)
	# preparing target
	df = df.iloc[:, 1:]
	df["cr"] = df.submissions / df.views
	df.drop(['views', 'submissions'], axis=1, inplace=True)

	# selecting almost constant features
	zero_percent = 100 - (df.astype(bool).sum(axis=0) * 100 / len(df))
	zero_value_df = pd.DataFrame({'column_name': df.columns,
								  'zero_percent': zero_percent})
	constant_95 = zero_value_df.loc[zero_value_df['zero_percent'] >= 95.0, 'column_name'].to_list()

	# synthetic features
	for c in constant_95:
		new_name = c + "_syn"
		df[new_name] = 0
		df.loc[df[c] > 0.0, new_name] = 1
	df.drop(constant_95, axis=1, inplace=True)

	# scaling
	columns_scale = df.columns[:30]
	for col in columns_scale:
		# the min max scaler requires a vector
		transformer = MinMaxScaler().fit(df[col].values.reshape(-1, 1)) # single feature
		transformed_data = transformer.transform(df[col].values.reshape(-1, 1))
		df[col+"_mm"] = transformed_data[:,0]
	# remove the original features of the scaled ones
	final_df = df.iloc[:,30:]

	to_delete = ['feat_18_syn', 'feat_31_syn', 'feat_39_syn', 'feat_32_syn',
				 'feat_29_mm', 'feat_14_mm', 'feat_05_mm', 'feat_43_mm', 'feat_28_mm',
				 'feat_01_mm', 'feat_42_mm', 'feat_46_mm', 'feat_45_mm', 'feat_25_mm']
	train_df = final_df.drop(to_delete, axis=1)
	more_to_delete = ['feat_19_mm', 'feat_38_mm', 'feat_36_mm', 'feat_40_mm', 'feat_47_mm']
	train_df.drop(more_to_delete, axis=1, inplace=True)
	train_df.to_pickle("../data/" + output_name)

def scale_num_features(feats_to_scale):
	'''
	Method to transform numerical features with the min-max transformation.
	We need to use the scaling from our training data on the testing data. Thus same scaling ranges.
	:param feats_to_scale: dictionary with the features and the values to scale
	:return: another dictionary with the same features (adjusted name) but the scaled values
	'''
	# apply minmax scaler to the input parameters according to the provided ranges
	min, max = [0, 1]
	scaled_feat = dict()
	#print("Keys in the dictionary {}".format(feats_to_scale.keys()))
	print("Scaling features")
	for feat in feats_to_scale.keys():
		try:
			value = feats_to_scale[feat]
			min_value = to_scale_ranges[feat][0]
			max_value = to_scale_ranges[feat][1]
			if value < min_value:
				value = min_value
			elif value > max_value:
				# if it is outside the range we assign the max accepted value
				value = max_value
			value_std = (value - min_value) / (max_value - min_value)
			value_scaled = value_std * (max - min) + min
			scaled_feat[feat+'_mm'] = value_scaled
		except Exception as e:
			print(e)
			print("Error scaling the feature {}".format(feat))
	return scaled_feat

def transform_syn_features(feats_to_transform):
	'''
	Method to transform the almost constant features into the synthetic binary ones
	:param feats_to_transform: dictionary with the features and their values
	:return: another dictionary with the equivalent synthetic features and the values
	'''
	syn_feat = dict()
	#print("Keys in the dictionary {}".format(feats_to_transform.keys()))
	print("Generating synthetic features")
	for feat in feats_to_transform.keys():
		if feats_to_transform[feat] > 0:
			syn_feat[feat+"_syn"] = 1
		else:
			syn_feat[feat+"_syn"] = 0
	return syn_feat

def group_features(all_features_dict):
	'''
	Given a dictionary with all features, it returns two groups of features:
		1. the group that needs scaling
		2. the group that needs to be transform into a synthetic binary feature
	:param all_features: a dictionary with all the features needed for the prediction
	:return: two different dictionaries: one for the features to scale and one for the features to syn
	'''
	feat_to_scale = dict()
	feat_to_syn = dict()
	print("Preparing two groups of features")
	#print("Keys in the dictionary {}".format(all_features_dict.keys()))
	for key in all_features_dict.keys():
		if key in constant_95:
			feat_to_syn[key] = all_features_dict[key]
		else:
			feat_to_scale[key] = all_features_dict[key]
	return feat_to_syn, feat_to_scale