import joblib 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest 

from data import load_data, basic_cleaning 
from features import build_features 

def main(): 
	
	print("Loading data... ") 

	df = load_data("data/raw/creditcard.csv")
	df = basic_cleaning(df)
	df = build_features(df) 

	X = df.drop(columns = ["Class"])
	y = df["Class"] 

	print("Spliting Datasets... ")

	X_train, X_test, y_train, y_test = train_test_split(
		X, 
		y, 
		test_size = 0.2, 
		stratify = y, 
		random_state = 42
	) 
	
	print(f"X_train: {X_train.shape}") 
	print(f"X_test: {X_test.shape}")
	
	print("Training Random Forest... ") 

	rf = RandomForestClassifier(
		n_estimators = 100, 
		max_depth = 10, 
		min_samples_leaf = 2, 
		class_weight = "balanced", 
		random_state = 42, 
		n_jobs = 1
	)

	rf.fit(X_train, y_train) 

	print("Training Isolation Forest... ") 

	X_train_normal = X_train[y_train == 0]

	iso = IsolationForest(
		n_estimators = 300, 
		contamination = 0.002, 
		random_state = 42, 
		n_jobs = 1
	) 

	iso.fit(X_train_normal) 

	print("Saving models... ") 

	joblib.dump(rf, "models/random_forest.joblib")
	joblib.dump(iso, "models/isolation_forest.joblib")
	joblib.dump(list(X_train.columns), "models/features.joblib")
	
	print("Training completed.")

if __name__ == "__main__":
	main() 


