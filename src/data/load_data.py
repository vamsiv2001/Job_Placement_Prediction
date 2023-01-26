import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_from_path(filepath):
    df = pd.read_csv(filepath)
    return df

def load_job_placement_data():
    input_path = "data/input_data/Job_Placement_Data.csv"
    df = load_data_from_path(input_path)
    return df

def save_preprocessed_data(df: pd.DataFrame):
    df.to_csv('data/preprocessed_data/preprocessed_data.csv', index=False) 
    return

def load_preprocessed_data():
    input_path = "data/preprocessed_data/preprocessed_data.csv"
    df = load_data_from_path(input_path)
    return df

def save_train_test_data(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, Xtest: pd.DataFrame, Ytest: pd.DataFrame):
    Xtrain.to_csv("data/training_data/training_data.csv", index=False)
    Ytrain.to_csv("data/training_data/training_data_result.csv", index=False)
    Xtest.to_csv("data/testing_data/testing_data.csv", index=False)
    Ytest.to_csv("data/testing_data/testing_data_result.csv", index=False)
    return

def split_train_test_data():

    df = load_preprocessed_data()
    X = df.drop(columns=["status"])
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    save_train_test_data(X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test
