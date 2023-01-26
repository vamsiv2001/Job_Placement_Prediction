from fixtures import data_preparation,logistic_regression_prediction
import pandas as pd
import pytest_check as check


def test_data_leak(data_preparation):
    xtrain, ytrain, xtest, ytest = data_preparation
    concat_df = pd.concat([xtrain, xtest])
    concat_df.drop_duplicates(inplace=True)
    assert concat_df.shape[0] == xtrain.shape[0] + xtest.shape[0]

def test_predicted_output_shape(logistic_regression_prediction):
    print("Logistic regression")
    xtest, ypred = logistic_regression_prediction
    check.equal(ypred.shape, (xtest.shape[0],1))
    #assert ypred.shape == (xtest.shape[0], 1)




