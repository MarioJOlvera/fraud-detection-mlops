import json 
import joblib 
import pandas as pd

from src.features import build_features 

MODEL_PATH = "models/random_forest.joblib"
FEATURES_PATH = "models/features.joblib" 

def load_artifacts(): 

	model = joblib.load(MODEL_PATH)
	expected_features = joblib.load(FEATURES_PATH) 

	return model, expected_features 


def prepare_input(data: dict) -> pd.DataFrame: 

	df = pd.DataFrame([data]) 
	df = build_features(df) 

	return df 

def align_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame: 

	for col in expected_features:
		if col not in df.columns:
			df[col] = 0

	return df[expected_features] 

def predict(data: dict) -> dict: 

	model, expected_features = load_artifacts()

	df = prepare_input(data)
	df = align_features(df, expected_features) 

	prediction = model.predict(df)[0]
	probability = model.predict_proba(df)[0][1] 

	return {
		"prediction": int(prediction), 
		"fraud_probability": float(probability)
	}



if __name__ == "__main__":
    sample_input = {
        "Time": 10000,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }

    result = predict(sample_input)
    print("Prediction result:")
    print(result)

	 
df = pd.read_csv("data/raw/creditcard.csv")

normal_sample = df[df["Class"] == 0].iloc[0].drop("Class").to_dict()
fraud_sample = df[df["Class"] == 1].iloc[0].drop("Class").to_dict()

print("Normal sample:")
print(predict(normal_sample))
print("==========================")
print("\nFraud sample: ") 
print(predict(fraud_sample)) 

