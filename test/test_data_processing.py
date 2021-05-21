import sys
from pathlib import Path
import pytest

cwd = Path.cwd()
main_dir = cwd.parent
sys.path.append(str(cwd))
sys.path.append(str(main_dir))

from data_processing import *

@pytest.fixture()
def get_sc_num_features():
	test_dict = {'feat_02': 2.0,
					'feat_06': 0.0,
					'feat_07': 6.0,
					'feat_08': 17,
					'feat_09': 100.0,
					'feat_11': 28.0,
					'feat_16': 2.0,
					'feat_17': 35.0,
					'feat_19': 10.0,
					'feat_21': 145.0,
					'feat_22': 101.0,
					'feat_23': 200.0,
					'feat_30': 7.0,
					'feat_33': 10.0,
					'feat_34': 30.0,
					'feat_36': 0.0,
					'feat_38': 99.0,
					'feat_40': 2.0,
					'feat_41': 100.0,
					'feat_47': 400.0}
	yield test_dict

@pytest.fixture()
def get_syn_num_features():
	test_dict = {'feat_04': 1.0,
				   'feat_20': 0.0,
				   'feat_26': 3.0,
				   'feat_31': 10.0,
				   'feat_32': 0.0,
				   'feat_15': 2.0,
				   'feat_35': 6.0,
				   'feat_10': 0.0,
				   'feat_12': 0.0,
				   'feat_13': 3.0,
				   'feat_24': 10.0,
				   'feat_37': 9.0,
				   'feat_03': 20.0,
				   'feat_39': 0.0,
				   'feat_44': 0.0,
				   'feat_27': 0.0,
				   'feat_18': 1.0}
	yield test_dict

class TestScaleNumFeatures:
	# TODO Check a key not present in the dictionary
	def test_scale_num_features(self, get_sc_num_features):
		out_dict = scale_num_features(get_sc_num_features)
		print(out_dict)
		assert len(out_dict) == 20
		assert round(out_dict["feat_02_mm"],3) == 0.036
		assert round(out_dict["feat_07_mm"],3) == 0.048
		# out of range value
		assert round(out_dict["feat_08_mm"],3) == 1.0



class TestTransformSynFeatures:
	# TODO Check a key not present in the dictionary
	def test_transform_syn_features(self, get_syn_num_features):
		out_dict = transform_syn_features(get_syn_num_features)
		assert len(out_dict) == 17
		assert out_dict["feat_04_syn"] == 1
		assert out_dict["feat_26_syn"] == 1
		assert out_dict["feat_32_syn"] == 0

class TestGroupFeatures:
	def test_group_features(self, get_sc_num_features, get_syn_num_features):
		all_feat_dict = get_sc_num_features.copy()
		all_feat_dict.update(get_syn_num_features)
		out_dict1, out_dict2 = group_features(all_feat_dict)
		assert len(out_dict1) == 17
		assert len(out_dict2) == 20