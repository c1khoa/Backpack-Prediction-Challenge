import src.data.make_dataset as data
import src.evaluation.evaluate_model as val
import src.features.build_features as ft
import src.models.train_model as model
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)
import os
MODEL_DIR = os.getenv("MODEL_DIR", "E:/Project/Backpack Prediction Challenge")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    logging.info(f"Using MODEL_DIR = {MODEL_DIR}")
    print("MODEL_DIR:", os.getenv("MODEL_DIR"))

    df = data.load_data(os.path.join(MODEL_DIR, "data/row\data.csv"))
    df = ft.build_feature(df)
    data.save_data(df, os.path.join(MODEL_DIR, "data\processed\data_preprocessed.csv"))

    df_train = data.load_data(os.path.join(MODEL_DIR, "data\processed\data_preprocessed.csv"))
    model.train_model(df_train)
    val.evaluate(df_train)

if __name__ == '__main__':
    main()