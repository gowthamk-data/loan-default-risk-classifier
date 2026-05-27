import os
import argparse
import joblib
import pandas as pd
import numpy as np

# LOAD + CLEAN 
def load_and_clean_data(filepath):
    """Load new data for the pipeline."""
    print(f"Loading new loan applications from {filepath}...")
    df = pd.read_csv(filepath) 
    
    user_ids = df['msisdn'].copy()
    df = df.drop('msisdn', axis=1)
    

    # 2. Date handling 
    df['pdate'] = pd.to_datetime(df['pdate'])
    df['pMonth'] = df['pdate'].dt.month
    df['pDay'] = df['pdate'].dt.day
    df['pWeekday'] = df['pdate'].dt.weekday
    df['isWeekend'] = (df['pdate'].dt.weekday >= 5).astype(int)
    df = df.drop('pdate', axis=1)

# 3. Ensure no leakage features sneaked into the production data
    leakage_features = ['payback30', 'payback90', 'label']
    df = df.drop(columns=leakage_features, errors='ignore')

    return df, user_ids

# PREDICTION ENGINE
def predict(input_data_path, output_data_path, model_path, threshold=0.35):
    
    # 1. Load and prep data
    X_new, user_ids = load_and_clean_data(input_data_path)
    
    # 2. Load the trained engine
    print(f"Loading production pipeline from {model_path}...")
    pipeline = joblib.load(model_path)
    
    # 3. Generate Probabilities
    print("Scoring new applications...")
    risk_probabilities = pipeline.predict_proba(X_new)[:, 1]
    
    # 4. Apply Threshold (0.35)
    print(f"Applying strict risk threshold of {threshold}...")
    predictions = (risk_probabilities >= threshold).astype(int)
    
    # 5. Format the Business Report
    results_df = pd.DataFrame({
        "Phone_Number_ID": user_ids,
        "Risk_Probability": np.round(risk_probabilities, 4),
        "Predicted_Class": predictions,
        "Action": np.where(predictions == 1, "Approve", "Flag for Review")
    })
    
    # 6. Save the results
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    results_df.to_csv(output_data_path, index=False)
    
    print(f"Success! Predictions generated for {len(results_df)} users.")
    print(f"Report saved to: {output_data_path}")

def engineer_features(X):
    """Feature engineering used during training."""

    X_out = X.copy()

    X_out["TotalLoanCount"] = (
        X_out["cnt_loans30"] +
        X_out["cnt_loans90"]
    )

    X_out["AvgLoanAmount30"] = np.where(
        X_out["cnt_loans30"] > 0,
        X_out["amnt_loans30"] /
        X_out["cnt_loans30"],
        0
    )

    X_out["AvgLoanAmount90"] = np.where(
        X_out["cnt_loans90"] > 0,
        X_out["amnt_loans90"] /
        X_out["cnt_loans90"],
        0
    )

    return X_out

# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict micro-credit loan defaults using the trained pipeline.")
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=r"../data/Credit-Project-Test-Data-file.csv", 
        help="Path to new loan applications CSV"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=r"../data/production_predictions.csv", 
        help="Path to save the final business report"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=r"../data/best_credit_pipeline.pkl",
        help="Path to the trained .pkl model"
    )
    
    # args=[] allows running safely in Jupyter Notebooks
    args = parser.parse_args(args=[]) 
    
    predict(
        input_data_path=args.input, 
        output_data_path=args.output,
        model_path=args.model
    )