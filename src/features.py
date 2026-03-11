import numpy as np 
import pandas as pd 

def build_features(df: pd.DataFrame) -> pd.DataFrame: 
	"""
	Feature Engineering
	"""

	df = df.copy()

	# log transformations 

	df["log_amount"] = np.log1p(df["Amount"])

	# hour proxy 
	
	df["hour_proxy"] = (df["Time"] // 3600) % 24 

	df["is_high_amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int) 

	return df 
