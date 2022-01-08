# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 18:05:15 2022

@author: akhil

"""

import joblib
import pandas as pd
import numpy as np

df = pd.read_csv("final_data.csv")
df.drop(columns=["Unnamed: 0"],inplace=True)

#splitting dataset into train, test and validate
X = df.drop(columns=["AverageTemperature"])
y = df["AverageTemperature"] 




def test_model():
     model = joblib.load("model.pkl")
    
     elements_pred =  model.predict(X)
     
     from sklearn.metrics import mean_squared_error
     error = np.sqrt(mean_squared_error(y,elements_pred))
     print(error)
     assert error < 9.0
     
