# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:18:44 2022

@author: akhil
"""

import re
import numpy as np
import pandas as pd
import json
from flask import Flask, request
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
std= joblib.load("scaler.save")
encoder = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()


df2 = pd.read_csv("data.csv")

encoder.fit(df2["City"].astype(str))
encoder2.fit(df2["dt"].astype(str))
encoder3.fit(df2["Country"].astype(str))

model = joblib.load("model.pkl")

@app.route('/', methods=['POST'])
def get_prediction():
        
    if request.method == 'POST':
        data = request.get_json()
        df= pd.json_normalize(data)
        df["AverageTemperature"] = 35.843
        df[["month","Latitude","Longitude","AverageTemperature","AverageTemperatureUncertainty","year"]] = std.transform(df[["month","Latitude","Longitude","AverageTemperature","AverageTemperatureUncertainty","year"]])
        df.drop(columns=["AverageTemperature"], inplace=True)
        X = df[["City", "Country","dt"]]
        print(df.shape)
        df["City"] = encoder.transform(df["City"].astype(str))
        df["dt"] = encoder2.transform(df["dt"].astype(str))
        df["Country"] = encoder3.transform(df["Country"].astype(str))
        print(df)
        prediction = model.predict(df)  # runs globally loaded model on the data
    return str(prediction)

@app.route("/", methods=["GET"])
def display():
        return "<h1>app</h1>

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    
    
#you have scaled the numerical inputs, now you have to merg the scaled inputs to the respectful data
#also instead of custom scaling try sklearn scaling too
