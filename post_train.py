from fixtures import data_preparation, return_model, logistic_regression_prediction
import pandas as pd
import numpy as np



def test_lr_invariance(return_model):
    model = return_model
    print("Checking for " + str(model.__class__.__name__))
    # Original status of sample1= '0'
    sample1 = [1,77.0,61.0,68.0,0,1]
    result_sample1= model.predict(np.array(sample1).reshape(1, -1))

    #Checking if Certain inputs keep the output  
    #Chaging Gender to male '0'
    sample1 = [0,77.0,61.0,68.0,0,1]
    result_sample1_gender= model.predict(np.array(sample1).reshape(1, -1))
    
    #Chaning Work Experience to No '1'
    sample = [1,77.0,61.0,68.0,1,1]
    result_sample1_Work_Exp= model.predict(np.array(sample1).reshape(1, -1))   

    #Changing Specialisation to Mkt&HR '0'
    sample1 = [1,77.0,61.0,68.0,0,0]
    result_sample1_Specialisation= model.predict(np.array(sample1).reshape(1, -1))
      
    assert result_sample1 == result_sample1_gender == result_sample1_Work_Exp == result_sample1_Specialisation



def test_lr_directional_expectation(return_model):
    model = return_model
    print("Checking for " + str(model.__class__.__name__))
    sample1 = [1,77.0,61.0,68.0,0,1]
    result_sample1= model.predict(np.array(sample1).reshape(1, -1))
    # Original status of sample1= '0'
    # Changing ssc board percentage from 77.0->54.0
    sample1 = [1,54.0,61.0,68.0,0,1] 
    result_sample1_ssc = model.predict(np.array(sample1).reshape(1,-1))
    
    #Changing hsc board percentage from 61.0->56.5
    sample1 = [1,77.0,56.5,68.0,0,1] 
    result_sample1_hsc = model.predict(np.array(sample1).reshape(1,-1)) 

    #Changing degree percentage from 68.0->44.0
    sample1 = [1,77.0,61.0,40.0,0,1] 
    result_sample1_degree = model.predict(np.array(sample1).reshape(1,-1))  
    #assert result_sample1 == result_sample1_degree
    assert result_sample1 == result_sample1_ssc == result_sample1_hsc == result_sample1_degree
    