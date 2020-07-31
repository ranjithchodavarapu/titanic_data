import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib .pyplot as plt
import math
import sklearn
from sklearn import linear_model
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



data=pd.read_csv('titanic.csv')
#print(data.head(10))

#x=data.isnull().sum()
#print(x)

sex=pd.get_dummies(data['Sex'],drop_first=True)
#print(sex)

pa=pd.get_dummies(data["Pclass"],drop_first=True)
#print(pa)

data=pd.concat([sex,data,pa],axis=1)
#print(data.head(5))

data.drop(['Pclass','Name','Sex'],axis=1,inplace=True)
#print(data.head(5))

x=data.drop("Survived",axis=1)
y=data['Survived']

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.3)
logmodel=linear_model.LogisticRegression()
logmodel.fit(x_train,y_train)



predictions=logmodel.predict(x_test)
s=classification_report(y_test,predictions)
print(s)


r=confusion_matrix(y_test,predictions)
print(r)

w=accuracy_score(y_test,predictions)
print(w)



