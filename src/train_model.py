import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

# LOAD + CLEAN DATA
def load_and_clean_data(filepath):
    """Load dataset and perform cleaning."""
    df = pd.read_csv(filepath).drop_duplicates()

    # Removes unique identifier
    df = df.drop('msisdn', axis=1, errors='ignore')

    # Date handling
    df['pdate'] = pd.to_datetime(df['pdate'])
    df['pMonth'] = df['pdate'].dt.month
    df['pDay'] = df['pdate'].dt.day
    df['pWeekday'] = df['pdate'].dt.weekday
    df['isWeekend'] = (df['pdate'].dt.weekday >= 5).astype(int)
    df = df.drop('pdate', axis=1)

    # Removes possible leakage features
    leakage_features = ['payback30', 'payback90']
    df = df.drop(columns=leakage_features, errors='ignore')

    return df

# FEATURE ENGINEERING
def engineer_features(X):
    X_out = X.copy()

    # Total loan behavior
    X_out["TotalLoanCount"] = X_out["cnt_loans30"] + X_out["cnt_loans90"]

    # Average loan amount
    X_out["AvgLoanAmount30"] = np.where(
        X_out["cnt_loans30"] > 0,
        X_out["amnt_loans30"] / X_out["cnt_loans30"],
        0
    )

    X_out["AvgLoanAmount90"] = np.where(
        X_out["cnt_loans90"] > 0,
        X_out["amnt_loans90"] / X_out["cnt_loans90"],
        0
    )

    return X_out

# PIPELINE
def build_pipeline():
    """Preprocessing + model pipeline."""
    # Feature engineering step
    feature_gen = FunctionTransformer(engineer_features)

    # Numeric transformer (No scaling needed for Random Forest)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Categorical transformer
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combined preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, make_column_selector(dtype_include=np.number)),
        ('cat', cat_transformer, make_column_selector(dtype_exclude=np.number))
    ])

    # ML pipeline
    pipeline = Pipeline(steps=[
        ('feature_gen', feature_gen),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',
            n_estimators=300,
            min_samples_split=10,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    return pipeline

# EVALUATION
def evaluate_model(model, X_val, y_val, threshold=0.35):
    """Evaluates trained model using a custom decision threshold."""
    # Probabilities
    y_prob = model.predict_proba(X_val)[:, 1]

    # Threshold tuning
    y_pred = (y_prob >= threshold).astype(int)

    # Print evaluation
    print("\nPRODUCTION MODEL EVALUATION")
    print("=" * 60)
    print(f"Threshold Used: {threshold}")
    print(f"\nROC-AUC Score: {roc_auc_score(y_val, y_prob):.4f}")
    print(f"PR-AUC Score:  {average_precision_score(y_val, y_prob):.4f}")
    print(f"Precision:     {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:        {recall_score(y_val, y_pred):.4f}")
    print(f"F1 Score:      {f1_score(y_val, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print("=" * 60 + "\n")

# FEATURE IMPORTANCE
def show_feature_importance(model_pipeline):
    """ Displays top feature importances dynamically extracted from the pipeline."""
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = model_pipeline.named_steps['classifier'].feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTOP 10 IMPORTANT FEATURES")
    print("=" * 60)
    print(importance_df.head(10))
    print("=" * 60 + "\n")

# MAIN
def main(input_path, output_model_path):
    # Load data
    df = load_and_clean_data(input_path)

    # Split features/target
    X = df.drop("label", axis=1)
    y = df["label"]

    # CROSS VALIDATION
    print("Running 5-Fold Cross Validation...")
    cv_pipeline = build_pipeline()
    cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    
    print(f"\nMean CV ROC-AUC: {cv_scores.mean():.4f}")
    print(f"CV Std Dev:      {cv_scores.std():.4f}")

    # TRAIN / VALIDATION SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Builds and trains final model
    print("\nTraining Final Model...")
    model_pipeline = build_pipeline()
    model_pipeline.fit(X_train, y_train)

    # Evaluates model
    evaluate_model(model_pipeline, X_val, y_val, threshold=0.35)

    # Feature importance
    show_feature_importance(model_pipeline)

    # SAVES MODEL
    output_dir = os.path.dirname(output_model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model_pipeline, output_model_path)
    print(f"\nModel successfully saved to:\n{output_model_path}")

# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Production-Grade Micro-credit Default Prediction Pipeline")
    parser.add_argument("--input", type=str, default=r"../data/Credit-Project-Train-Data-file.csv", help="Path to training data")
    parser.add_argument("--output", type=str, default=r"../models/best_credit_pipeline.pkl", help="Path to save model")
    
    # args=[] allows running in Jupyter securely without crashing
    args = parser.parse_args(args=[]) 
    main(input_path=args.input, output_model_path=args.output)