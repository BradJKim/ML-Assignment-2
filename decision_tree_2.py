#-------------------------------------------------------------------------
# AUTHOR: Brad Kim
# FILENAME: decision_tree_2.py
# SPECIFICATION: Decision Tree for different data sets
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
        
    """ functions to convert feature into numbers """
    #Age,Spectacle Prescription,Astigmatism,Tear Production Rate,Recommended Lenses
    
    def convertAge(x):
        match x:
            case 'Young':           return 1
            case 'Prepresbyopic':   return 2
            case 'Presbyopic':      return 3
            case _:                 return 1
            
    def convertSpec(x):
        match x:
            case 'Hypermetrope':    return 1
            case 'Myope':           return 2
            case _:                 return 1
            
    def convertAst(x):
        match x:
            case 'Yes':             return 1
            case 'No':              return 2
            case _:                 return 1
            
    def convertTPR(x):
        match x:
            case 'Normal':          return 1
            case 'Reduced':         return 2
            case _:                 return 1
            
    def convertRec(x):
        match x:
            case 'Yes':             return 1
            case 'No':              return 2
            case _:                 return 1

    # index to feature conversion mapping
    convertFunc = {
        0: convertAge,
        1: convertSpec,
        2: convertAst,
        3: convertTPR,
        4: convertRec
    }
    
    for row in dbTraining:
        updated_feat_row = []
    
        for i,feat in enumerate(row[:-1]):
            changed_val = convertFunc[i](feat)
            updated_feat_row.append(changed_val)
        
        X.append(updated_feat_row)

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    
    for row in dbTraining:
        updated_class = []
        
        changed_val = convertFunc[len(row)-1](row[-1])
        updated_class.append(changed_val)
        
        Y.append(updated_class)

    accuracy = 0
    iterations = 10
    
    #Loop your training and test tasks 10 times here
    for i in range (iterations):

        #Fitting the decision tree to the data setting max_depth=5
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        #--> add your Python code here
        
        dbTest = []
        
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
                
            updated_feat_row = []
            updated_class = []

            for i,feat in enumerate(row[:-1]):
                changed_val = convertFunc[i](feat)
                updated_feat_row.append(changed_val)
            
            changed_val = convertFunc[len(row)-1](row[-1])
            updated_class.append(changed_val)
                
            class_predicted = [int(clf.predict([updated_feat_row])[0])]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            
            if(class_predicted[0] == 1):
                if (updated_class[0] == class_predicted[0]):
                    TP+=1
                else:
                    FP+=1
            else:
                if (updated_class[0] == class_predicted[0]):
                    TN+=1
                else:
                    FN+=1
         
        accuracy += (TP + TN) / (TP + TN + FP + FN)

        
    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    
    accuracy /= iterations


    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    
    print(f"Final accuracy when training on {ds}: {accuracy}")