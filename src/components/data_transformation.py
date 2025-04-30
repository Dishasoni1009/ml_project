import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This just to create all my pickle file which is responsible to converting categorical features into numerical . 
        if you want to perform standard scaler.
        Creates and returns the preprocessing pipeline for the loan approval prediction system.
        """
        try:
            numerical_columns = [
                'ApplicantIncome',
                'CoapplicantIncome',
                'LoanAmount',
                'Loan_Amount_Term',
                'Credit_History',
                'Dependents'
            ]

            categorical_columns = [
                'Gender',
                'Married',
                'Education',
                'Self_Employed',
                'Property_Area'
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  #pipeline for handling missing values
                    ("scaler", StandardScaler())  # pipeline for doing standard scaler
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), #pipeline for handling missing values 
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))

                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Transforms the training and testing datasets using the preprocessing pipeline.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded training and testing datasets.")


            # Drop Loan_ID column if present
            if 'Loan_ID' in train_df.columns:
                train_df.drop('Loan_ID', axis=1, inplace=True)
            if 'Loan_ID' in test_df.columns:
                test_df.drop('Loan_ID', axis=1, inplace=True)
           

            target_column_name = "Loan_Status"

            # Encode target labels
            label_encoder = LabelEncoder()
            train_df[target_column_name] = label_encoder.fit_transform(train_df[target_column_name])
            test_df[target_column_name] = label_encoder.transform(test_df[target_column_name])

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessing object saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
