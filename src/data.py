import pandas as pd 

def load_data(path: str) -> pd.DataFrame: 
	"""
	Load Dataset from CSV
	"""

	df = pd.read_csv(path) 
	
	return df 

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame: 
	"""
	Basic data cleaning
	"""

	df = df.drop_duplicates().copy()
	
	return df 


