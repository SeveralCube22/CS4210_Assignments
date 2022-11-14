#-------------------------------------------------------------------------
# AUTHOR: Viswadeep Manam
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
from math import inf
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training [:, :64] first elem slice by rows (i.e select all rows), second elem slice by col (i.e select 0-63 columns)
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

max_percept_accuracy = -inf
max_mlp_accuracy = -inf

for w in n: #iterates over n

    for b in r: #iterates over r

        for a in range(2): #iterates over the algorithms

            #Create a Neural Network classifier
            if a==0:
               clf = Perceptron(eta0=w, shuffle=b, max_iter=1000) #eta0 = learning rate, shuffle = shuffle the training data
            else:
               clf = MLPClassifier(activation='logistic', learning_rate_init=w, hidden_layer_sizes=(25,), shuffle =b, max_iter=1000) #learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer, shuffle = shuffle the training data

            #Fit the Neural Network to the training data
            correct_pred, total_pred = 0, 0
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            for (x_testSample, y_testSample) in zip(X_test, y_test):
               pred = clf.predict([x_testSample])[0]
               correct_pred += 1 if pred == y_testSample else 0
               total_pred += 1
            
            accuracy = correct_pred / total_pred
             
            print_str = "Highest Perceptron " if a == 0 else "Highest MLP "
            print_str += f"accuracy so far: {accuracy}, Parameters: learning rate={w}, shuffle={b}"
            
            if a == 0 and accuracy > max_percept_accuracy:
               print(print_str)
               max_percept_accuracy = accuracy
            
            elif accuracy > max_mlp_accuracy:
               print(print_str)
               max_mlp_accuracy = accuracy
                  











