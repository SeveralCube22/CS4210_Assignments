#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
import csv
from optparse import Values
from sklearn.naive_bayes import GaussianNB

#reading the training data in a csv file
#--> add your Python code here

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

def populate_Values(db, headers): 
    VALUES = [{} for i in range(0, len(headers))]
    for i, header in enumerate(headers):
        counter = 1
        for row in db:
            value = row[i]
            if value not in VALUES[i]:
                VALUES[i][value] = counter
                counter += 1
    return VALUES
    
def populate_X_Y(db, headers, exclude_data_headers=["Day"], exclude_label_headers=[]): 
    X = []
    Y = []

    x_labels = []
    y_labels = []
    
    for row in db:
        x_sample, x_label = [], []
        y_sample, y_label = 0, 0
        
        
        for i, header in enumerate(headers):
            
            if header not in exclude_data_headers:
                if i < len(headers) - 1:
                    x_sample.append(VALUES[i][row[i]])
                else:
                    y_sample = VALUES[i][row[i]]
                    
            if header not in exclude_label_headers:
                if i < len(headers) - 1:
                    x_label.append(row[i])
                else:
                    y_label = row[i]
    
        X.append(x_sample)
        Y.append(y_sample)
        x_labels.append(x_label)
        y_labels.append(y_label)
    
    return X,Y, x_labels, y_labels

db = []
headers = []
with open("weather_training.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                db.append(row)
            else:
                headers = row

VALUES = populate_Values(db, headers)           
X, Y, _, _ = populate_X_Y(db, headers)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTest = []
headersTest = []
with open("weather_test.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTest.append(row)
            else:
                headersTest = row



#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

xTest, _, xLabels, _ = populate_X_Y(dbTest, headersTest, exclude_data_headers=["Day", "PlayTennis"])

for i in range(0, len(xTest)):
    pred = clf.predict_proba([xTest[i]])[0]
    
    if pred[0] >= 0.75:
        print(xLabels[i][0].ljust(15), xLabels[i][1].ljust(15), xLabels[i][2].ljust(15), xLabels[i][3].ljust(15), xLabels[i][4].ljust(15), "YES".ljust(15) + " %.2f\n" % (pred[0]))
    elif pred[1] >= 0.75:
       print(xLabels[i][0].ljust(15), xLabels[i][1].ljust(15), xLabels[i][2].ljust(15), xLabels[i][3].ljust(15), xLabels[i][4].ljust(15), "NO".ljust(15) + "%.2f\n" % (pred[1]))