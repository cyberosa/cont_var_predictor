# features that were found to be in the training dataset 95% constant
constant_95 = ['feat_04',
			   'feat_20',
			   'feat_26',
			   'feat_31',
			   'feat_32',
			   'feat_15',
			   'feat_35',
			   'feat_10',
			   'feat_12',
			   'feat_13',
			   'feat_24',
			   'feat_37',
			   'feat_03',
			   'feat_39',
			   'feat_44',
			   'feat_27',
			   'feat_18']

# input features that need scaling
to_scale_ranges = {'feat_02': [0.0, 55.0],
					 'feat_06': [0.0, 96.0],
					 'feat_07': [0.0, 124.0],
					 'feat_08': [0.0, 15.0],
					 'feat_09': [0.0, 165.0],
					 'feat_11': [0.0, 38.0],
					 'feat_16': [0.0, 137.0],
					 'feat_17': [0.0, 135.0],
					 'feat_21': [0.0, 236.0],
					 'feat_22': [0.0, 134.0],
					 'feat_23': [0.0, 205.0],
					 'feat_30': [0.0, 124.0],
					 'feat_33': [0.0, 105.0],
					 'feat_34': [0.0, 96.0],
					 'feat_41': [0.0, 135.0]}

# app parameters, with the names we expect to receive them from the API
app_param_cols = ['feat_04', 'feat_20', 'feat_26', 'feat_15',
       'feat_35', 'feat_10', 'feat_12', 'feat_13',
       'feat_24', 'feat_37', 'feat_03', 'feat_44',
       'feat_27', 'feat_02', 'feat_06', 'feat_07',
	   'feat_08', 'feat_09', 'feat_11', 'feat_16',
	   'feat_17', 'feat_21', 'feat_22', 'feat_23',
	   'feat_30', 'feat_33', 'feat_34', 'feat_41']

# input features for the model
# first the synthetic features and next the scaled
input_model_cols = ['feat_04_syn', 'feat_20_syn', 'feat_26_syn', 'feat_15_syn',
       'feat_35_syn', 'feat_10_syn', 'feat_12_syn', 'feat_13_syn',
       'feat_24_syn', 'feat_37_syn', 'feat_03_syn', 'feat_44_syn',
       'feat_27_syn', 'feat_02_mm', 'feat_06_mm', 'feat_07_mm',
	   'feat_08_mm', 'feat_09_mm', 'feat_11_mm', 'feat_16_mm',
	   'feat_17_mm', 'feat_21_mm', 'feat_22_mm', 'feat_23_mm',
	   'feat_30_mm', 'feat_33_mm', 'feat_34_mm', 'feat_41_mm']