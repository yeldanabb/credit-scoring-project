from fastapi import FastAPI
import pandas as pd
import scorecardpy as sc
import joblib
import pickle
import os

app = FastAPI(title="Credit Scoring API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'credit_scoring_lr_model.pkl')
features_path = os.path.join(BASE_DIR, 'models', 'features_list.pkl')
bins_path = os.path.join(BASE_DIR, 'models', 'bins.pkl')

model = joblib.load(model_path)
features_list = joblib.load(features_path)
with open(bins_path, 'rb') as f:
    bins = pickle.load(f)

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])
    if 'term' in input_df.columns and input_df['term'].dtype == 'object':
        input_df['term_int'] = input_df['term'].str.replace(' months', '').astype(int)
    
    input_df['installment_to_inc'] = (input_df['installment'] * 12) / input_df['annual_inc']
    input_df['loan_to_inc'] = input_df['loan_amnt'] / input_df['annual_inc']
    
    input_woe = sc.woebin_ply(input_df, bins)
    
    available_columns = [col for col in features_list if col in input_woe.columns]
    X = input_woe[available_columns]
    
    prob = model.predict_proba(X)[:, 1][0]
    decision = "Reject" if prob > 0.25 else "Approve"
    
    return {
        "probability_of_default": round(float(prob), 4),
        "decision": decision,
        "score": int((1 - prob) * 1000) 
    }