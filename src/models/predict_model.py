import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
MODEL_DIR = os.getenv("MODEL_DIR", "E:/Project/Backpack Prediction Challenge")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_model(model_path):
    logging.info(f"Load model from: {model_path}")
    return joblib.load(model_path)

def predict(model, X_test):
    logging.info("Đang dự đoán dữ liệu test...")
    try:
        return model.predict(X_test)
    except Exception as e:
        logging.error(f"Lỗi khi dự đoán: {e}")
        return None
