import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer 
dataset=load_breast_cancer(as_frame=True) 
X=dataset['data'] 
y=dataset['target'] 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) 
from sklearn.preprocessing import StandardScaler 
ss_train=StandardScaler() 
X_train=ss_train.fit_transform(X_train) 
ss_test=StandardScaler() 
X_test=ss_test.fit_transform(X_test) 
models={} 
# Logistic Regression 
from sklearn.linear_model import LogisticRegression 
models['Logistic Regression']=LogisticRegression() 
# Support Vector Machines 
from sklearn.svm import LinearSVC 
models['Support Vector Machines']=LinearSVC() 
# Decision Trees 
from sklearn.tree import DecisionTreeClassifier 
models['Decision Trees']=DecisionTreeClassifier() 
# Random Forest 
from sklearn.ensemble import RandomForestClassifier 
models['Random Forest']=RandomForestClassifier() 
# Naive Bayes 
from sklearn.naive_bayes import GaussianNB 
models['Naive Bayes']=GaussianNB() 
# K-Nearest Neighbors 
from sklearn.neighbors import KNeighborsClassifier 
models['K-Nearest Neighbor']=KNeighborsClassifier() 
from sklearn.metrics import accuracy_score,precision_score,recall_score 
accuracy,precision,recall={},{},{} 
for key in models.keys(): 
  models[key].fit(X_train,y_train) 
  predictions=models[key].predict(X_test) 
  accuracy[key]=accuracy_score(predictions,y_test) 
  precision[key]=precision_score(predictions,y_test) 
  recall[key]=recall_score(predictions,y_test) 
import pandas as pd 
df_model=pd.DataFrame(index=models.keys(),columns=['Accuracy','Precision','Recall']) 
df_model['Accuracy']=accuracy.values() 
df_model['Precision']=precision.values() 
df_model['Recall']=recall.values() 
df_model
ax=df_model.plot.barh() 
ax.legend( 
    ncol=len(models.keys()), 
    bbox_to_anchor=(0,1), 
    loc="lower left", 
    prop={'size':14} 
) 
plt.tight_layout()
from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test,predictions) 
TN,FP,FN,TP=confusion_matrix(y_test,predictions).ravel() 
print("True Positive(TP):",TP) 
print("False Positive(FP):",FP) 
print("True Negative(TN):",TN) 
print("False Negative(FN):",FN) 
accuracy=(TP+TN)/(TP+FP+TN+FN) 
print("Accuracy of the binary classifier = {:0.3f}".format(accuracy))