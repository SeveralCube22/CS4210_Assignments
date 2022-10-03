#-------------------------------------------------------------------------
# AUTHOR: Viswadeep Manam
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

def determine_categorical_value(attrib):
    if attrib == "Young" or attrib == "Myope" or attrib == "Yes" or attrib == "Reduced": return 1
    elif attrib == "Prepresbyopic" or attrib == "Hypermetrope" or attrib == "No" or attrib == "Normal": return 2
    elif attrib == "Presbyopic": return 3
    else: return -1

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append(row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    X = [[determine_categorical_value(attrib) for attrib in row[0:len(row)-1]] for row in dbTraining] # Get row values from 0 to row-2, ignoring the Y or label values

    Y = [determine_categorical_value(row[-1]) for row in dbTraining]

    lowest_accuracy = 1.0
    
    #loop your training and test tasks 10 times here
    for i in range (10):
        
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)
                
        correct_preds = 0
        total_preds = 0
        
        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
           
            test_X = [[determine_categorical_value(attrib) for attrib in data[0:len(data)-1]]]
            actual_Y = determine_categorical_value(data[-1])
           
            class_predicted = clf.predict(test_X)[0]
           
            if class_predicted == actual_Y:
                correct_preds += 1
               
            total_preds += 1


        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        accuracy = correct_preds / total_preds
        if accuracy < lowest_accuracy:
            lowest_accuracy = accuracy

    #print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("final accuracy on %s: %f" %(ds, lowest_accuracy))



