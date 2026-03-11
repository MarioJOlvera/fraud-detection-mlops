import json 
import joblib 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	precision_score, 
	recall_score, 
	f1_score, 
	roc_auc_score, 
	average_precision_score, 
	confusion_matrix
) 

from data import load_data, basic_cleaning
from features import build_features

def evaluate_random_forest(model, X_test, y_test): 

	"""
	Evaluate Random Forest usign classification metrics
	"""

	rf_probs = model.predict_proba(X_test)[:, 1] 
	rf_preds = (rf_probs >= 0.5).astype(int)

	metrics = {
		"precision": precision_score(y_test, rf_preds), 
		"recall": recall_score(y_test, rf_preds),
		"f1": f1_score(y_test, rf_preds), 
		"roc_auc": roc_auc_score(y_test, rf_preds), 
		"pr_auc": average_precision_score(y_test, rf_preds), 
		"confusion_matrix": confusion_matrix(y_test, rf_preds).tolist()
	}

	return metrics 

def evaluate_isolation_forest(model, X_test, y_test):

	"""
	Evaluate Isolation Forest by mapping:
	1 -> normal -> 0
	-1 -> anomaly -> 1
	"""

	iso_raw_preds = model.predict(X_test)
	iso_preds = (iso_raw_preds == -1).astype(int)

	metrics = {
		"precision": precision_score(y_test, iso_preds, zero_division = 0),
		"recall": recall_score(y_test, iso_preds, zero_division = 0),
		"f1": f1_score(y_test, iso_preds, zero_division = 0),
		"confusion_matrix": confusion_matrix(y_test, iso_preds).tolist()
	}


	return metrics 

def main(): 
	print("Loading datasets... ") 

	df = load_data("data/raw/creditcard.csv")
	df = basic_cleaning(df) 
	df = build_features(df) 

	X = df.drop(columns = ["Class"]) 
	y = df["Class"]

	print("Cleaning train/test split... ") 

	X_train, X_test, y_train, y_test = train_test_split(
		X, 
		y, 
		test_size = 0.2, 
		stratify = y, 
		random_state = 42
	) 

	print("Loading models... ") 

	rf = joblib.load("models/random_forest.joblib") 
	iso = joblib.load("models/isolation_forest.joblib")
	feature_columns = joblib.load("models/features.joblib") 

	X_test = X_test[feature_columns] 

	print("Evaluating Random Forest...") 

	rf_metrics = evaluate_random_forest(rf, X_test, y_test) 
	
	print("Evaluating Isolation Forest...") 
	iso_metrics = evaluate_isolation_forest(iso, X_test, y_test) 

	all_metrics = {
		"random_forest": rf_metrics, 
		"isolation_forest": iso_metrics
	}

	print("\n== Evaluation Results ==") 
	print(json.dumps(all_metrics, indent = 4)) 

	with open("models/metrics.json", "w") as f: 
		json.dump(all_metrics, f, indent = 4) 

	print("\nMetrics saved to models/metrics.json")


if __name__ == "__main__":
	main() 
		
