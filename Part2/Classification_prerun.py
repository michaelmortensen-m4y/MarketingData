# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:27:58 2020

@author: anjaeh
"""

from os import getcwd
import pandas as pd
import numpy as np
#Enable python to go into parent folder to open helper.py file
import sys
sys.path.insert(1,'..')

from helper import *
from Classification_models import *
from Part1.SummaryStatistics import ordinals, nominals
import matplotlib.pyplot as plt
#import sklearn.linear_model as lm
#imports for decision tree and graphviz plotting
from sklearn import tree, model_selection
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
#from platform import system
#from toolbox_02450 import windows_graphviz_call
from matplotlib.image import imread



# Load marketing data to get attributes names, a pandas dataframe object and a numpy array
attNames, marketingData_pd, marketingData_np = loadMarketingData()
attNamesEncoded, marketingDataEncoded_pd, marketingDataEncoded_np = encodeCategorical(marketingData_pd, ordinals=ordinals, nominals=nominals)


#Extract outcome vector y, convert to np
classLabels = marketingData_np[:,-1] 
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.asarray([classDict[value] for value in classLabels])

# Compute values of C --> number of classes
C = len(classNames)

#full dataset with 1-out-of-K encoded categorial variables
#drop outcome variable and duration
drop = ['y', 'duration']
marketingDataEncoded_pd = marketingDataEncoded_pd.drop(drop, axis = 1)
marketingDataEncoded_np = marketingDataEncoded_pd.to_numpy()

for d in range(len(drop)):
    attNamesEncoded.remove(drop[d])


N, M = marketingDataEncoded_np.shape
# Standardize the data

means = marketingDataEncoded_np.mean(axis=0) # get mean of each column
X_full = marketingDataEncoded_np - np.ones((N,1))*means # Get matrix X by substracting the mean from each value in the marketingdata
X_full = X_full*(1/np.std(X_full,0)) #Deviding by standard deviation to normalize

N, M = X_full.shape

# Add offset attribute
#X_full = np.concatenate((np.ones((X_full.shape[0],1)),X_full),1)
#attNamesEncoded = [u'Offset']+attNamesEncoded
#M = M+1

## Outer Crossvalidation
# Create crossvalidation partition for evaluation
 
K = 5
CV_outer = model_selection.KFold(K, shuffle=True)


# Values of lambda
#lambdas = np.power(10.,range(-5,9))
#lambda_opt_all = np.empty((K,2))
#lambda_opt_lasso = np.empty((K,1))

#Paramters for tree
splits = np.asarray([1, 20])
depth = np.asarray([10, 100])
criterion =['gini', 'entropy']


#nearest neighbor
neighbor = np.arange(1,5)

#Initialize errors
# Final error rates stored for the table

Error_test = np.empty((K,10))
#Other Error measures
False_Positive_Rate = np.empty((K,10))
True_Positive_Rate = np.empty((K,10))
#Intermediate calculated for model selection
Error_gen = np.empty((1, 10))

k=0
for train_index, test_index in CV_outer.split(X_full,y):

    # extract training and test set for current CV fold
    X_train = X_full[train_index]
    y_train = y[train_index]
    X_test = X_full[test_index]
    y_test = y[test_index]
    N_par, M_par = X_train.shape
    N_test, M_test = X_test.shape
    #CV_inner = model_selection.KFold(K, shuffle = True)
    
    #baseline model
    #y_train_est = baseline_classification(y_train, y_train)
    y_test_est = baseline_classification(y_train, y_test)
    #Error_train_base[k] = errorRate(y_train_est, y_train)
    Error_test[k,0], True_Positive_Rate[k,0], False_Positive_Rate[k,0] = errorRate(y_test_est, y_test)
    
    
   # validation_error_log = np.empty((K,len(lambdas)))
    #validation_error_lasso = np.empty((K,len(lambdas)))  
    
    dist=2
    metric = 'minkowski'
    metric_params = {} # no parameters needed for minkowski
    
    # You can set the metric argument to 'cosine' to determine the cosine distance
    #metric = 'cosine' 
    #metric_params = {} # no parameters needed for cosine
    
    # To use a mahalonobis distance, we need to input the covariance matrix, too:
    #metric='mahalanobis'
    #metric_params={'V': np.cov(X_train, rowvar=False)}
    
    # Fit classifier and classify the test points
    for n,i in zip(neighbor, range(1,5)):
        knclassifier = KNeighborsClassifier(n_neighbors=n, p=dist, 
                                        metric=metric,
                                        metric_params=metric_params)
        knclassifier.fit(X_train, y_train)
        y_est = knclassifier.predict(X_test)
        Error_test[k,i], True_Positive_Rate[k,i], False_Positive_Rate[k,i] = errorRate(y_est, y_test)
    


    
    dtc_3 = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=10, max_depth = 40)
    dtc_3 = dtc_3.fit(X_train,y_train)
    y_est = dtc_3.predict(X_test)
    Error_test[k,5], True_Positive_Rate[k,5], False_Positive_Rate[k,5] = errorRate(y_est, y_test)
    
    
   

    k+=1

for n in range(0, 6):
    Error_gen[0,n] =  np.sum(N_test/N*Error_test[:,n])


    # Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plt.plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plt.plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
plt.title('Synthetic data classification - KNN');

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
plt.figure(2);
plt.imshow(cm, cmap='binary', interpolation='None');
plt.colorbar()
plt.xticks(range(C)); plt.yticks(range(C));
plt.xlabel('Predicted class'); plt.ylabel('Actual class');
plt.title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

plt.show()
    
  
    
      
    
    
