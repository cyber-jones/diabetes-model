import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model():
    try:

        # load teh content from .env file to env vars
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")       

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        logging.info("Training script started")  # noqa: LOG015
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape {df.shape}")  # noqa: LOG015

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced"))
            ]
        )
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")  # noqa: LOG015

        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))

        train_report = classification_report(y_train, pipeline.predict(X_train))
        test_report = classification_report(y_test, pipeline.predict(X_test))

        logging.info(f"Train Accuracy: {train_acc:.3f}")  # noqa: LOG015
        logging.info(f"Test Accuracy: {test_acc:.3f}")  # noqa: LOG015

        logging.info(f"Train Classification Report: \n{train_report}")  # noqa: LOG015
        logging.info(f"Test Classification Report: \n{test_report}")  # noqa: LOG015

        dump(pipeline, MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")  # noqa: LOG015

        logging.info("Training script finished")  # noqa: LOG015

    
    except Exception as e:
        print(f"Training failed: {e}")
        logging.exception(f"Training script failed: {e}")  # noqa: LOG015, TRY401
        raise


if __name__ == "__main__":
    train_model()
    