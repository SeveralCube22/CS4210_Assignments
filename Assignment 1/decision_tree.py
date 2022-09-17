#-------------------------------------------------------------------------
# AUTHOR: Viswadeep Manam
# FILENAME: Decision Tree
# SPECIFICATION: An implementation of the Decision Tree classifier
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
def determine_categorical_value(attrib):
  if attrib == "Young" or attrib == "Myope" or attrib == "Yes" or attrib == "Reduced": return 1
  elif attrib == "Prepresbyopic" or attrib == "Hypermetrope" or attrib == "No" or attrib == "Normal": return 2
  elif attrib == "Presbyopic": return 3
  else: return -1

X = [[determine_categorical_value(attrib) for attrib in row[0:len(row)-1]] for row in db] # Get row values from 0 to row-2, ignoring the Y or label values
#transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
Y = [determine_categorical_value(row[-1]) for row in db]

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy', )
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()