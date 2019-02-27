import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn import tree

col_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Type']

df=pd.read_csv("glass.csv", header=None,names=col_names)
df.shape

print(df.head())



feature = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
x=df[feature]
# y is target values
y=df.Type

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test


from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini",max_depth=5)
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Missclassified :",(y_test!=y_pred).sum())
print("Accuracy:",accuracy_score(y_test, y_pred))

max_dep=[]
for i in range(1,100):
    clf = DecisionTreeClassifier(criterion="gini",max_depth=i)
    clf = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    a=accuracy_score(y_test, y_pred)
    max_dep.append(a)

print(max_dep,end=" ")    
print("\nBest Aaccuracy")    
print(max(max_dep))    

