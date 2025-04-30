import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    """
    Configuration for Data Ingestion.
    Contains paths for saving raw, training, and testing datasets.
    """
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    """
    Handles loading, preprocessing, and splitting the loan dataset.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def load_data(self, file_path):
        """
        Load the dataset from a CSV file.
        """
        try:
            logging.info(f"Loading dataset from {file_path}")
            df = pd.read_csv(file_path)
            logging.info(f"Dataset loaded successfully with shape {df.shape}")
            return df
        except Exception as e:
            raise CustomException(f"Failed to load data: {e}", sys)

    def preprocess_data(self, df):
        """
        Perform basic preprocessing (optional here based on your EDA).
        """
        try:
            logging.info("Starting basic preprocessing")
            # Example: drop rows with missing target
            if 'Loans' in df.columns:
                df = df.dropna(subset=['Loans'])

            # You can add more cleaning logic here if needed
            logging.info("Preprocessing complete")
            return df
        except Exception as e:
            raise CustomException(f"Preprocessing error: {e}", sys)

    def initiate_data_ingestion(self, csv_file_path):
        """
        Orchestrates the data ingestion: loading, preprocessing, splitting, and saving.
        """
        try:
            df = self.load_data(csv_file_path)
            df = self.preprocess_data(df)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info(f"Train and test sets saved to {self.ingestion_config.train_data_path} and {self.ingestion_config.test_data_path}")

            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    csv_file_path = "/home/disha-soni/Desktop/project/ml_project/notebook/data/loans.csv"
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion(csv_file_path)

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(
        ingestion.ingestion_config.train_data_path,
        ingestion.ingestion_config.test_data_path
    )

    