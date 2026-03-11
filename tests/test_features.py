import pandas as pd 

from src.features import build_features

def test_build_features_creates_expected_columns(): 

	df = pd.DataFrame({
		"Time": [0, 3600, 7200], 
		"Amount": [10.0, 100.0, 1000.0],
		"Class": [0,1,0]
	}) 

	result = build_features(df) 

	assert "log_amount" in result.columns 
	assert "hour_proxy" in result.columns
	assert "is_high_amount" in result.columns 


def test_hour_proxy_is_computed_correctly():

	df = pd.DataFrame({
		"Time": [0, 3600, 7200],
		"Amount": [10.0, 100.0, 1000.0],
		"Class": [0, 1, 0]
	}) 

	result = build_features(df) 

	assert result["hour_proxy"].tolist() == [0, 1, 2] 
