#-------------------------------------------------------------------------
# AUTHOR: Brad Kim
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes for email classification dataset
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate
import csv

#Reading the training data in a csv file
#--> add your Python code here

dbTraining = []
X = []
Y = []

with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTraining.append (row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

def convertOut(x):
    match x:
        case 'Sunny':           return 1
        case 'Overcast':        return 2
        case 'Rain':            return 3
        case _:                 return 1

def convertTemp(x):
    match x:
        case 'Cool':            return 1
        case 'Mild':            return 2
        case 'Hot':             return 3
        case _:                 return 1
        
def convertHum(x):
    match x:
        case 'Normal':          return 1
        case 'High':            return 2
        case _:                 return 1
        
def convertWind(x):
    match x:
        case 'Strong':          return 1
        case 'Weak':            return 2
        case _:                 return 1
        
def convertPlay(x):
    match x:
        case 'Yes':             return 1
        case 'No':              return 2
        case _:                 return 1

convertFunc = {
    1: convertOut,
    2: convertTemp,
    3: convertHum,
    4: convertWind,
    5: convertPlay
}

for row in dbTraining:
    updated_feat_row = []

    for i,feat in enumerate(row[:-1]):
        if i != 0:
            changed_val = convertFunc[i](feat)
            updated_feat_row.append(changed_val)
        
    X.append(updated_feat_row)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

for row in dbTraining:
    changed_val = convertFunc[len(row)-1](row[-1])
    Y.append(changed_val)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here

dbTest = []

with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTest.append(row)

#Printing the header of the solution
#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

matrix = []

for row in dbTest:
    feat_row = []

    for i,feat in enumerate(row[:-1]):
        if i != 0:
            changed_val = convertFunc[i](feat)
            feat_row.append(changed_val)
                
    class_prob = clf.predict_proba([feat_row])[0]
    yes_prob = class_prob[0]
    no_prob = class_prob[1]
    
    class_pred = 'Yes' if (yes_prob >= no_prob) else 'No'
    class_prob = yes_prob if (yes_prob >= no_prob) else no_prob
    
    if class_prob > 0.75:
        matrix.append([row[0], row[1], row[2], row[3], row[4], class_pred, class_prob])
            
print(tabulate(matrix, headers=["Day", "Outlook", "Temp", "Humidity", "Wind", "Prediction", "Probability"], tablefmt="grid"))