from src.data.load_data import split_train_test_data
from src.data.preprocessing_data import preprocessing_data
from src.model.lr_train import logistic_regression
from src.model.lr_predict import predict_on_test_data

import pytest
import pytest_check as check
import pandas as pd



@pytest.fixture
def data_preparation():
    preprocessing_data()
    return split_train_test_data()


@pytest.fixture
def logistic_regression_prediction(data_preparation):
    xtrain, ytrain, xtest, ytest = data_preparation
    lr = logistic_regression(xtrain, ytrain)
    ypred = predict_on_test_data(lr, xtest)
    return xtest, ypred 

@pytest.fixture
def return_model(data_preparation):
    xtrain, ytrain, xtest, ytest = data_preparation
    lr = logistic_regression(xtrain, ytrain)
    return lr
