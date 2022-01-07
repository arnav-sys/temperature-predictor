# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:41:38 2022

@author: akhil
"""

import pandas as pd
import numpy as np

df = pd.read_csv("final_data.csv")

#splitting dataset into train, test and validate
from sklearn.model_selection import train_test_split
X = df.drop(columns=["AverageTemperature"])
y = df["AverageTemperature"] 
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=20)
X_train, X_Validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=20)

#model building(linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

#making predictions(linear regression)
predictions = lr.predict(X_Validate)

#evaluating the model
from sklearn.metrics import mean_squared_error
print("(linear regression)root mean squared error is: ", np.sqrt(mean_squared_error(y_validate,predictions)))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(lr, X_train, y_train, cv=3)
print(scores.mean())


#model bubilding(lasso regression)
from sklearn.linear_model import Lasso
lar = Lasso()
lar.fit(X_train, y_train)

#making predictions(lasso regression)
predictions = lar.predict(X_Validate)


#evaluating the model
print("(lasso regression)root mean squared error is: ", np.sqrt(mean_squared_error(y_validate,predictions)))

scores = cross_val_score(lar, X_train, y_train, cv=3)
print(scores.mean())


#model building(knn)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(X_train,y_train)

#making predictions(knn)
predictions = knn.predict(X_Validate)
print("(knn)root mean squared error is: ", np.sqrt(mean_squared_error(y_validate,predictions)))

scores = cross_val_score(knn, X_train, y_train, cv=3)
print(scores.mean())


#all the models were doing great but we choose to go with linear regression

#parameter optimization
import optuna

def objective(trial):
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    
    normalize = trial.suggest_categorical("normalize", [True, False])
    
    n_jobs = trial.suggest_int("n_jobs",1,2,3)
    
    lr = LinearRegression(fit_intercept, normalize, n_jobs)
    
    return cross_val_score(lr, X_train, y_train, n_jobs=-1, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

trial = study.best_trial

print("Best parameters: {}".format(trial.params))

model = LinearRegression(fit_intercept=True, normalize=True, n_jobs= 1)
model.fit(X_train, y_train)

#making final predictions
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print("root mean squared error is: ", np.sqrt(mean_squared_error(y_test,predictions)))

#saving model
import joblib
joblib.dump(model, "model.pkl")