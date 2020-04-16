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


##### Added later for training of final model with lambda = 10 after selection and error estimation ######
#Final model on whole data set and extract parameters
#Fit intercept was set false since offset attribute was added to the datamatrix X_full
#Otherwise the model has a w0 coef around 0 and another parameter intercept_ which is -2.5
lambda_final = 10
log_reg = lm.LogisticRegression(penalty='l2', C=1/lambda_final, max_iter = 300, fit_intercept = False )
log_reg.fit(X_full, y)

#parameter vector
w_0 = log_reg.intercept_
w_vector = log_reg.coef_
w_full = pd.DataFrame(w_vector.T, index = attNamesEncoded, columns = ['w'])
w_select = w_full[(w_full['w'] < -0.1) | (w_full['w'] > 0.1)]

##################################################


## Outer Crossvalidation
# Create crossvalidation partition for evaluation
random_seed = 3404
K = 10
CV_outer = model_selection.KFold(K, shuffle=True, random_state = random_seed)


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
Error_gen_balanced = np.empty((1, len(lambdas)))
w_log = np.empty((M,K,len(lambdas)))
w_balanced = np.empty((M,K,len(lambdas)))
y_est_full = np.empty((1,3))
y_true = []

k=0
for train_index, test_index in CV_outer.split(X_full,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K)) 

    # extract training and test set for current CV fold
    X_train = X_full[train_index]
    y_train = y[train_index]
    X_test = X_full[test_index]
    y_test = y[test_index]
    N_par, M_par = X_train.shape
    CV_inner = model_selection.KFold(K, shuffle = True, random_state = random_seed)
    
    y_est_cv_fold = []
    #baseline model
    #y_train_est = baseline_classification(y_train, y_train)
    y_test_est = baseline_classification(y_train, y_test)
    #Error_train_base[k] = errorRate(y_train_est, y_train)
    Error_test[k,0], True_Positive_Rate[k,0], False_Positive_Rate[k,0] = errorRate(y_test_est, y_test)
    
    y_est_cv_fold.append(y_test_est)
    y_true.append(y_test)
    
    validation_error_log = np.empty((K,len(lambdas)))
    validation_error_balanced = np.empty((K,len(lambdas)))  
    j = 0
    for train_inner, test_inner in CV_inner.split(X_train, y_train):
        # extract training and test set for current CV fold
        print('\nInner Crossvalidation fold: {0}/{1}'.format(j+1,K))
        X_train_in = X_train[train_inner]
        y_train_in = y_train[train_inner]
        X_test_in = X_train[test_inner]
        y_test_in = y_train[test_inner]
                     
        
        N_val, M_val = X_test_in.shape
        #logistic regression
        for n in range(0, len(lambdas)):
            print('\nTraining log Model: {0}'.format(n+1))
            log_reg = lm.LogisticRegression(penalty='l2', C=1/lambdas[n], max_iter = 300 )
            log_reg.fit(X_train_in, y_train_in)

            #y_train_est = log_reg.predict(X_train_in).T
            y_test_est = log_reg.predict(X_test_in).T
            w_log[:,j,n] = log_reg.coef_.squeeze()
            
            #Error_train_LogReg[k] = errorRate(y_train_est, y_train)
            validation_error_log[j,n], temp1, temp2 = errorRate(y_test_est, y_test_in)
            
        for n in range(0, len(lambdas)):
            print('\nTraining balanced Model: {0}'.format(n+1))
            log_reg = lm.LogisticRegression(penalty='l2', C=1/lambdas[n], max_iter = 300, class_weight = 'balanced' )
            log_reg.fit(X_train_in, y_train_in)

            #y_train_est = log_reg.predict(X_train_in).T
            y_test_est = log_reg.predict(X_test_in).T
            w_balanced[:,j,n] = log_reg.coef_.squeeze()
            
            #Error_train_LogReg[k] = errorRate(y_train_est, y_train)
            validation_error_balanced[j,n], temp1, temp2 = errorRate(y_test_est, y_test_in)
        
        
        
               
        j +=1
    ### Back to outer loop for training of selected model and test error estimation ###
    
    #Calculate generelization error for model selection 
    for n in range(0, len(lambdas)):
        Error_gen_log[0,n] =  np.sum(N_val/N_par*validation_error_log[:,n])
    
    #min_error = np.min(Error_gen_log)
    #opt_lambda_idx = np.argmin(Error_gen_log)
    lambda_opt_all[k,0] = lambdas[np.argmin(Error_gen_log)]
    
    #Train LogReg model with selected lambda on outer training data loop
    print('Training selected model')
    log_reg = lm.LogisticRegression(penalty='l2', C=1/lambda_opt_all[k,0], max_iter =300)
    log_reg.fit(X_train, y_train)
    y_est = log_reg.predict(X_test).T
    
    y_est_cv_fold.append(y_est)
    
    #y_est_full.append(y_est)
    #Error rate for optimal logReg model for outer loop
    Error_test[k,1], True_Positive_Rate[k,1], False_Positive_Rate[k,1] = errorRate(y_est, y_test)
    
    #Calculate generelization error for model selection lasso:
    for n in range(0, len(lambdas)):
        Error_gen_balanced[0,n] =  np.sum(N_val/N_par*validation_error_balanced[:,n])
    
    lambda_opt_all[k,1] = lambdas[np.argmin(Error_gen_balanced)]
    
    
    #Train balanced model with selected lambda on outer training data loop
       
    log_balanced = lm.LogisticRegression(penalty='l2', C=1/lambda_opt_all[k,1], max_iter =300,
                                         class_weight = 'balanced')                                  
    log_balanced.fit(X_train, y_train)
    y_est = log_balanced.predict(X_test).T
     #Error rate for optimal Lasso model for outer loop
    
    y_est_cv_fold.append(y_est)
    
    Error_test[k,2], True_Positive_Rate[k,2], False_Positive_Rate[k,2] = errorRate(y_est, y_test)
    
    
    #treat estimations from the three models as columns 
    y_est_cv_fold = np.stack(y_est_cv_fold, axis=1)
    #add the estimations for all models from this cv to the full list
    
    y_est_full = np.concatenate((y_est_full, y_est_cv_fold), axis = 0)
    
    #Make figure with lambda for the second crossvalidation fold
    
    if k == 1:
        #calculate mean of coefficients vs. lambda matrix
        mean_w_vs_lambda_log = np.squeeze(np.mean(w_log,axis=1))
        mean_w_vs_lambda_balanced = np.squeeze(np.mean(w_balanced, axis = 1))
        
        plt.figure(1, figsize=(16,8))
        plt.title('Regularized logistic regression')
        plt.subplot(1,2,1)
        plt.semilogx(lambdas,mean_w_vs_lambda_log.T,'.-') # Don't plot the bias term
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean Coefficient Values')
        
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        plt.subplot(1,2,2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(lambdas[np.argmin(Error_gen_log)])))
        plt.loglog(lambdas,Error_gen_log[0,:],'r.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Squared error (crossvalidation)')
        plt.yticks(np.power(10.,range(-2,-1)))
        plt.legend(['Validation error'])
        plt.grid()
        
        plt.figure(2, figsize=(16,8))
        plt.title('Regularized logistic regression')
        plt.subplot(1,2,1)
        plt.semilogx(lambdas,mean_w_vs_lambda_balanced.T,'.-') # Don't plot the bias term
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean Coefficient Values')
        
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        plt.subplot(1,2,2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(lambdas[np.argmin(Error_gen_balanced)])))
        plt.loglog(lambdas,Error_gen_balanced[0,:],'r.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Squared error (crossvalidation)')
        plt.yticks(np.power(10.,range(-2,-1)))
        plt.legend(['Validation error'])
        plt.grid()
    
    k+=1

y_est_full = y_est_full[1:,:]
#y_est_full = np.concatenate(y_est_full)
y_true = np.concatenate(y_true)





#transfer to dataframe and export as csv

#frame.to_csv(out_path, index = False)
    

