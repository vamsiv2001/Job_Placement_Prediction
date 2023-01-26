import pandas as pd


def predict_on_test_data(model, xtest):
    y_test = model.predict(xtest)
    filename = str(model.__class__.__name__)+"_testing_data_results.csv"
    prediction = pd.DataFrame(y_test)
    pd.DataFrame(y_test).to_csv("data/final_result/"+filename)
    return prediction

