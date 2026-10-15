import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")

LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")
        
def predict(input_data: dict) -> int:
    
    model = load(MODEL_PATH)
    df = pd.DataFrame([input_data])
    predicted_value = model.predict(df)[0]
    return predicted_value


def main():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )
        
        logging.info("Model loaded")  # noqa: LOG015

        input_data = {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 80,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.45,
            "Age": 35
        }
        
        prediction = predict(input_data=input_data)

        if prediction == 1:
            print("⚠️ Model predicts: Diabetes")
            logging.info("Prediction result: Diabetes")  # noqa: LOG015
        else:
            print("✅ Model predicts: No Diabetes")
            logging.info("Prediction result: No Diabetes")  # noqa: LOG015

        logging.info("Prediction script finished")  # noqa: LOG015


    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        logging.exception(f"Prediction script failed: {e}")  # noqa: LOG015, TRY401
        raise


if __name__ == "__main__":
    main()
