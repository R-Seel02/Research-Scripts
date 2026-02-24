import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import pypots as pots
import random
import csv
import pandas as pd 
from sklearn.model_selection import KFold
#from sklearn import svm as SVM
from libsvm.svmutil import *

inputData = pd.read_csv('raw_data_with_gender.csv', sep=',', ) #replace with desired file 

kf = KFold(n_splits=10, shuffle=True, random_state=42) #10 fold cross validation

X = inputData[['days_trend_mood0', 'days_trend_anxiety0']].values
Y = inputData


for fold, (train_index, test_index) in enumerate(kf.split(inputData)):
    print(f"Fold {fold+1}")
    X_train, X_test = X.iloc[train_index], X[test_index]
    y_train, y_test = inputData['target_variable'].iloc[train_index], inputData['target_variable'].iloc[test_index] #replace 'target_variable' with the name of the target variable in your dataset
    
    # Train your model here using X_train and y_train
    model = SVM() #replace with desired model
    model.fit(X_train, y_train)
    
    # Evaluate your model here using X_test and y_test
    predictions = model.predict(X_test)
    accuracy = sk.metrics.accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

