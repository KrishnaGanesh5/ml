import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder 
iris_data = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/ghpages/data/iris.csv") 
print("Original dataset:") 
print(iris_data.head()) 
# Remove duplicates 
iris_data_no_duplicates = iris_data.drop_duplicates() 
# Split dataset into features and target variable 
X = iris_data_no_duplicates.drop(columns=['species']) 
y = iris_data_no_duplicates['species'] 
# Convert categorical labels into numerical values 
label_encoder = LabelEncoder() 
y = label_encoder.fit_transform(y) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
# Calculate bias and variance 
bias = np.mean((y_test - y_pred) ** 2) 
variance = np.mean(np.var(y_pred, axis=0)) 
print("Bias:", bias) 
print("Variance:", variance) 
# Cross-validation 
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy') 
accuracy_cv = scores.mean() 
print("Accuracy using cross-validation:", accuracy_cv)