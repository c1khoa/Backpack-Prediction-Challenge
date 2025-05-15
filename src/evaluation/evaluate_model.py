from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import joblib
import logging
logging.basicConfig(level=logging.INFO)
import os
from src.models.predict_model import load_model
MODEL_DIR = os.getenv("MODEL_DIR", "E:/Project/Backpack Prediction Challenge")
os.makedirs(MODEL_DIR, exist_ok=True)

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def k_fold_report(model, model_name, X, y, cv = 5):

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = {
        'rmse': make_scorer(rmse_score)
    }

    kfold_result = cross_validate(model, X, y, cv=kfold, scoring=scoring, return_train_score=False)

    df = pd.DataFrame({
        'fit_time': kfold_result['fit_time'],
        'score_time': kfold_result['score_time'],
        'rmse': kfold_result['test_rmse'],
    })

    df.loc['mean'] = df.mean()
    df.index = [f'kfold_{i+1}' for i in range(cv)] + ['mean']

    df = df.round(4)
    df.to_csv(os.path.join(MODEL_DIR, f"reports/k_fold/{model_name}_kfold_report.csv"))

def evaluate(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    logging.info("Saving evaluate of Linear Regression in ./models/lr_kfold_report.csv")
    lr = load_model(os.path.join(MODEL_DIR, "models/lr_model.pkl"))
    k_fold_report(lr, 'lr', X, y)

    logging.info("Saving evaluate of XGBoost in ./models/xgb_kfold_report.csv")
    xgb = load_model(os.path.join(MODEL_DIR, "models/xgb_model.pkl"))
    k_fold_report(xgb, 'xgb', X, y)

    logging.info("Saving evaluate of LightGBM in ./models/lgbm_kfold_report.csv")
    lgbm = load_model(os.path.join(MODEL_DIR, "models/lgbm_model.pkl"))
    k_fold_report(lgbm, 'lgbm', X, y)

    logging.info("All models are evaluated!!")