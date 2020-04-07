# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:20:56 2020

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
import sklearn.linear_model as lm
#imports for decision tree and graphviz plotting
from sklearn import tree, model_selection
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
X_full = np.concatenate((np.ones((X_full.shape[0],1)),X_full),1)
attNamesEncoded = [u'Offset']+attNamesEncoded
M = M+1

## Outer Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV_outer = model_selection.KFold(K, shuffle=True)


# Values of lambda
lambdas = np.power(10.,range(-5,9))
lambda_opt_all = np.empty((K,2))
#lambda_opt_lasso = np.empty((K,1))

#Initialize errors
# Final error rates stored for the table
#Error_train_base = np.empty((K,1))
#Error_test_base = np.empty((K,1))
#Error_train_LogReg = np.empty((K,1))
#Error_test_LogReg = np.empty((K,1))
Error_test = np.empty((K,3))
False_Positive_Rate = np.empty((K,3))
True_Positive_Rate = np.empty((K,3))
Error_gen_log = np.empty((1, len(lambdas)))

k=0
for train_index, test_index in CV_outer.split(X_full,y):

    # extract training and test set for current CV fold
    X_train = X_full[train_index]
    y_train = y[train_index]
    X_test = X_full[test_index]
    y_test = y[test_index]
    N_par, M_par = X_train.shape
    CV_inner = model_selection.KFold(K, shuffle = True)
    
    #baseline model
    #y_train_est = baseline_classification(y_train, y_train)
    y_test_est = baseline_classification(y_train, y_test)
    #Error_train_base[k] = errorRate(y_train_est, y_train)
    Error_test[k,0], True_Positive_Rate[k,0], False_Positive_Rate[k,0] = errorRate(y_test_est, y_test)
    
    
    validation_error_log = np.empty((K,len(lambdas)))
    validation_error_lasso = np.empty((K,len(lambdas)))  
    j = 0
    for train_inner, test_inner in CV_inner.split(X_train, y_train):
        # extract training and test set for current CV fold
        X_train_in = X_train[train_inner]
        y_train_in = y_train[train_inner]
        X_test_in = X_train[test_inner]
        y_test_in = y_train[test_inner]
                     
        
        N_val, M_val = X_test_in.shape
        #logistic regression
        for n in range(0, len(lambdas)):
            
            log_reg = lm.LogisticRegression(penalty='l2', C=1/lambdas[n], max_iter = 200 )
            log_reg.fit(X_train_in, y_train_in)

            #y_train_est = log_reg.predict(X_train_in).T
            y_test_est = log_reg.predict(X_test_in).T
            
            #Error_train_LogReg[k] = errorRate(y_train_est, y_train)
            validation_error_log[j,n], temp1, temp2 = errorRate(y_test_est, y_test_in)
            
        
        
        #Compare with lasso regression to deselect unimportant featurs
        #inner_error_rate = np.empty((len(lambdas),1)) 
        for n in range(0, len(lambdas)):
            
            log_lasso = lm.LogisticRegression(penalty='l1', C=1/lambdas[n], max_iter = 200,
                                            solver = 'liblinear')
            log_lasso.fit(X_train_in, y_train_in)

            #y_train_est = log_reg.predict(X_train_in).T
            y_test_est = log_lasso.predict(X_test_in).T
            
            #Error_train_LogReg[k] = errorRate(y_train_est, y_train)
            validation_error_lasso[j,n], temp1, temp2 = errorRate(y_test_est, y_test_in)
        
               
        j +=1
    ### Back to outer loop for training of selected model and test error estimation ###
    
    #Calculate generelization error for model selection 
    for n in range(0, len(lambdas)):
        Error_gen_log[0,n] =  np.sum(N_val/N_par*validation_error_log[:,n])
    
    #min_error = np.min(Error_gen_log)
    #opt_lambda_idx = np.argmin(Error_gen_log)
    lambda_opt_all[k,0] = lambdas[np.argmin(Error_gen_log)]
    
    #Train LogReg model with selected lambda on outer training data loop
    
    log_reg = lm.LogisticRegression(penalty='l2', C=1/lambda_opt_all[k,0], max_iter =200)
    log_reg.fit(X_train, y_train)
    y_est = log_reg.predict(X_test).T
    
          
    #Error rate for optimal logReg model for outer loop
    Error_test[k,1], True_Positive_Rate[k,1], False_Positive_Rate[k,1] = errorRate(y_est, y_test)
    
    #Calculate generelization error for model selection lasso:
    for n in range(0, len(lambdas)):
        Error_gen_log[0,n] =  np.sum(N_val/N_par*validation_error_lasso[:,n])
    
    lambda_opt_all[k,1] = lambdas[np.argmin(Error_gen_log)]
    
    
    #Train Lasso model with selected lambda on outer training data loop
       
    log_lasso = lm.LogisticRegression(penalty='l1', C=1/lambda_opt_all[k,1], max_iter =200,
                                      solver = 'liblinear')
    log_lasso.fit(X_train, y_train)
    y_est = log_lasso.predict(X_test).T
     #Error rate for optimal Lasso model for outer loop
    Error_test[k,2], True_Positive_Rate[k,2], False_Positive_Rate[k,2] = errorRate(y_est, y_test)
    
    
       
    #Visualize prediction of logReg for each outer fold
    f = plt.figure();
    class0_ids = np.nonzero(y_test==0)[0].tolist()
    
    class1_ids = np.nonzero(y_test==1)[0].tolist()
    class0 = plt.plot(class0_ids, y_est[class0_ids], 'xb', label = 'no')
    class1 = plt.plot(class1_ids, y_est[class1_ids], 'xr', label = 'yes')
    
    plt.xlabel('Customer'); plt.ylabel('Predicted prob. of "yes"');
    plt.legend()
    plt.ylim(-0.01,1.5)

    plt.show()
    
    k+=1
    
    
