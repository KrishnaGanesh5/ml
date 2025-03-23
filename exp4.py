import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
IceCream=pd.read_csv('/content/IceCreamData.csv') 
print(IceCream)
# Divide the data into “Attributes” and “labels” 
X = IceCream[['Temperature']] 
y = IceCream['Revenue'] 
# Split 80% of the data to the training set while 20% of the data to test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
# Create a Linear Regression model and fit it 
regressor =LinearRegression(fit_intercept=True) 
regressor.fit(X_train,y_train) 
print('Linear Model Coeff (m) =' , regressor.coef_) 
print('Linear Model Coeff (b) =' , regressor.intercept_) 
 
# Predicting the data 
y_predict=regressor.predict(X_test) 
print(y_predict) 
 
# Scatter plot on Training Data 
plt.scatter(X_train,y_train,color='blue') 
plt.plot(X_train,regressor.predict(X_train),color='red') 
plt.ylabel('Revenue [$]') 
plt.xlabel('Temperatur [degC]') 
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand (Training)') 
 
# Scatter plot on Testing Data 
plt.scatter(X_test,y_test,color='blue') 
plt.plot(X_test,regressor.predict(X_test),color='red') 
plt.ylabel('Revenue [$]') 
plt.xlabel('Temperatur [degC]') 
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand (Training)') 
 
# Prediction the revenve using Temperature Value directly 
print('---------0---------') 
Temp = -0 
Revenue = regressor.predict([[Temp]]) 
print(Revenue) 
print('--------35----------') 
Temp = 35 
Revenue = regressor.predict([[Temp]]) 
print(Revenue) 
print('--------55----------') 
Temp = 55 
Revenue = regressor.predict([[Temp]]) 
print(Revenue)