import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

diamonds= pd.read_csv("diamonds.csv")
replace_dict_cut = {"Fair":1,"Good":2, "Very Good":3, "Premium":4, "Ideal":5}
diamonds["cut"] = diamonds["cut"].map(replace_dict_cut)
replace_dict_color = {"J":1, "I":2, "H":3, "G":4, "F":5, "E":6, "D":7}
diamonds["color"] =diamonds["color"].map(replace_dict_color)
replace_dict_clarity = {"I1":1,"SI2":2, "SI1":3, "VS2":4, "VS1":5, "VVS2":6, "VVS1":7, "IF":8}
diamonds["clarity"]=diamonds["clarity"].map(replace_dict_clarity)


col=["cut", "color", "clarity", "price"]
for a in col:
    diamonds[a]=diamonds[a].astype(float)

diamonds.drop('x', axis='columns', inplace=True)
diamonds.drop('y', axis='columns', inplace=True)
diamonds.drop('z', axis='columns', inplace=True)

x=diamonds[['carat','color','cut']]
y=diamonds['price']
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=7)

model = RandomForestRegressor(max_depth=5,  random_state=12, n_estimators = 1000)
ml_fit = model.fit(x_train, y_train)
y_pred_rf = ml_fit.predict(x_test)

import pickle
f = open('model.pkl', 'wb')
pickle.dump(model, f)
f.close()
