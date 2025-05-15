from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import logging
logging.basicConfig(level=logging.INFO)
import os

MODEL_DIR = os.getenv("MODEL_DIR", "E:/Project/Backpack Prediction Challenge")
os.makedirs(MODEL_DIR, exist_ok=True)

def linear_regression(X, y):
    lr = LinearRegression()
    lr.fit(X, y)
    joblib.dump(lr, os.path.join(MODEL_DIR, "models/lr_model.pkl"))

def xgboost(X, y):
    xgb = XGBRegressor(n_estimators=993, 
                       max_depth=4, 
                       learning_rate=0.028556044417328296, 
                       subsample=0.8312536468844265, 
                       colsample_bytree=0.6326239524981074, 
                       gamma=1.8794617120838553, 
                       reg_alpha=3.784852065072265, 
                       reg_lambda=4.585180571781613)
    xgb.fit(X, y)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "models/xgb_model.pkl"))

def lightgbm(X, y):
    lgbm = LGBMRegressor(n_estimators=768, 
                         max_depth=3, 
                         learning_rate=0.010111043605656992, 
                         num_leaves=2253, 
                         feature_fraction=0.8558569654112169, 
                         bagging_fraction=0.6135842843593063, 
                         bagging_freq=2, 
                         lambda_l1=4.229608547745549, 
                         lambda_l2=0.010761843865784326,
                         verbose=-1)
    lgbm.fit(X, y)
    joblib.dump(lgbm, os.path.join(MODEL_DIR, "models/lgbm_model.pkl"))


def train_model(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    logging.info("Saving model Linear Regression...")
    linear_regression(X, y)
    logging.info("Saving model XGBoost...")
    xgboost(X, y)
    logging.info("Saving model LightGBM...")
    lightgbm(X, y)
    logging.info("All models saved successfully!!")