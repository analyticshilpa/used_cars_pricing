# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using the training dataset and evaluates using the test dataset. Saves trained model.
"""
import mlflow
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path

mlflow.start_run()  # Start the MLflow experiment run

#this text is to change the train.py file to trigger the workflow for Project 3 for Great Learning
def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of tress in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read train and test data from CSV
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")

    # Split the data into input(X) and output(y)
    y_train = train_df['price']
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # Initialize and train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict using the Random Forest Regressor on test data
    yhat_test = model.predict(X_test)

    # Compute and log accuracy score
    mse = mean_squared_error(y_test, yhat_test)
    print(f'Mean Squared Error of Random Forest Regressor on test set: {mse:.2f}')
    # Logging the MSE score as a metric
    mlflow.log_metric("MSE", float(mse))

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
