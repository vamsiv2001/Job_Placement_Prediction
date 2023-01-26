from sklearn.linear_model import LogisticRegression

def logistic_regression(xtrain, ytrain):
    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    return lr
