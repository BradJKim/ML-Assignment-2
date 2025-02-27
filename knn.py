#-------------------------------------------------------------------------
# AUTHOR: Brad Kim
# FILENAME: knn.py
# SPECIFICATION: KNN for email classification dataset
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

total_error = 0
it = 0

#Loop your data to allow each instance to be your test set
for row in db:
    it += 1
    
    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    
    X = []
    for record in db:
        v = []
        
        if record != row:
            for i, j in enumerate(record):
                if i != len(row) -1 :
                    v.append(float(j))
                    
            X.append(v)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    
    Y = []
    
    for record in db:
        if record != row:
            Y.append(float(0 if record[-1] == 'ham' else 1))

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    
    testSample = []
    
    for i, j in enumerate(row):
        if i != len(row)-1:
            testSample.append(float(j))
                        
    true_label = float(0 if row[-1] == 'ham' else 1)
    
    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    
    res = class_predicted == true_label
    if not res: total_error+=1
    
    print(f"Prediction is accurate for i={it}: {res}")


#Print the error rate
#--> add your Python code here

avg_error = total_error / it
print(f"Avg. Error Rate: {avg_error}")




