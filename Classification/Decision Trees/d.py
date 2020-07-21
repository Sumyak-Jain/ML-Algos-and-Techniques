import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

my_data = pd.read_csv("https://raw.githubusercontent.com/sumyak/ML-Algos-and-Techniques/master/Classification/Decision%20Trees/drug200.csv", delimiter=",")

#Pre-processing
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 


y = my_data["Drug"]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#Modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)

#Prediction
predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

#Evaluation
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
