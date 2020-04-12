# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:11:27 2020

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
from sklearn import model_selection

from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
plt.rcParams.update({'font.size': 12})



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


#Set up crossvalidation
random_seed = 3404
K = 10
CV = model_selection.KFold(K,shuffle=True, random_state = random_seed)

# Values of lambda
hidden_units = np.asarray((2,3,5,7,10,30,60))
hidden_opt_all = np.empty((K,1))
hidden_opt_tpr = np.empty((K,1))
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
Error_gen_ANN = np.empty((1, len(hidden_units)))
True_Pos_Gen = np.empty((1, len(hidden_units)))

# Make figure for holding summaries (errors and learning curves)
#summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
#color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
#              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model structure
#n_hidden_units = 100 # number of hidden units in the signle hidden layer
#n_hidden_2 = 10
# The lambda-syntax defines an anonymous function, which is used here to 
# make it easy to make new networks within each cross validation fold
n_hidden_units = hidden_units[0]
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                          
                    #torch.nn.ReLU(),
                    #torch.nn.Linear(n_hidden_units, n_hidden_2),
                    #torch.nn.Tanh(),
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
# Since we're training a neural network for binary classification, we use a 
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.BCELoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the 
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 10000
print('Training model of type:\n{}\n'.format(str(model())))

errors = [] # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the 
# indices to be used for training and testing in each split, and calling 
# the enumerate-method with this simply returns this indices along with 
# a counter k:
for k, (train_index, test_index) in enumerate(CV.split(X_full,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, 
    # and convert them to PyTorch tensor
    #added squeeze to avoid error about different output/input sizes
    X_train = torch.Tensor(X_full[train_index,:] )
    y_train = torch.Tensor(y[train_index] ).squeeze()
    X_test = torch.Tensor(X_full[test_index,:] )
    y_test = torch.Tensor(y[test_index] ).squeeze()
    
    CV_inner = model_selection.KFold(K, shuffle = True, random_state = random_seed)
    
    #baseline model
    #y_train_est = baseline_classification(y_train, y_train)
    y_test_np, y_train_np = y_test.data.numpy(), y_train.data.numpy()
    y_test_est = baseline_classification(y_train_np, y_test_np)
    #Error_train_base[k] = errorRate(y_train_est, y_train)
    Error_test[k,0], True_Positive_Rate[k,0], False_Positive_Rate[k,0] = errorRate(y_test_est, y_test_np)
    
    
    validation_error_ANN = np.empty((K,len(hidden_units)))
    True_Positive_Val = np.empty((K,len(hidden_units)))
    
    j = 0
    for train_inner, test_inner in CV_inner.split(X_train, y_train):
        print('\nInner Crossvalidation fold: {0}/{1}'.format(j+1,K))
        # extract training and test set for current CV fold
        X_train_in = X_train[train_inner]
        y_train_in = y_train[train_inner]
        X_test_in = X_train[test_inner]
        y_test_in = y_train[test_inner]
                     
        
        N_val, M_val = X_test_in.shape
    
        # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
        # and see how the network is trained (search for 'def train_neural_net',
        # which is the place the function below is defined)
        for n in range(0, len(hidden_units)):
            print('\nTraining Model: {0}'.format(n+1))
            n_hidden_units = hidden_units[n]
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_in,
                                                               y=y_train_in,
                                                               n_replicates=1,
                                                               max_iter=max_iter)
            
        
            print('\n\tBest loss: {}\n'.format(final_loss))
        
            # Determine estimated class labels for test set
            y_sigmoid = net(X_test_in) # activation of final note, i.e. prediction of network
            y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8).data.numpy().squeeze() # threshold output of sigmoidal function
            #y_test = y_test.type(dtype=torch.uint8)
            # Determine errors and error rate
            #e = (y_test_est != y_test_in)
            #error_rate = (sum(e).type(torch.float)/len(y_test_in)).data.numpy()
            y_test_in_np = y_test_in.data.numpy()
            validation_error_ANN[j,n], True_Positive_Val[j,n], temp2= errorRate(y_test_est, y_test_in_np)
            
            #errors.append(error_rate) # store error rate for current CV fold 
            
        j+=1
    ### Back to outer loop for training of selected model and test error estimation ###
    
    #Calculate generelization error for model selection 
    for n in range(0, len(hidden_units)):
        Error_gen_ANN[0,n] =  np.mean(validation_error_ANN[:,n])
        True_Pos_Gen[0,n] = np.mean(True_Positive_Val[:,n])
    
    
    
    #min_error = np.min(Error_gen_log)
    #opt_lambda_idx = np.argmin(Error_gen_log)
    hidden_opt_all[k,0] = hidden_units[np.argmin(Error_gen_ANN)]
    
    #Select via true positive rate
    hidden_opt_tpr[k,0] = hidden_units[np.argmax(True_Pos_Gen)]
    
    #Train ANN model with selected hidden units on outer training data loop
    print('Training selected model')
    n_hidden_units = hidden_opt_all[k,0].astype(int)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=1,
                                                       max_iter=max_iter)
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8).data.numpy().squeeze() # threshold output of sigmoidal function
    #y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    #e = (y_test_est != y_test)
    #error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    #Error_test[k,1] = error_rate
    y_test_np = y_test.data.numpy()
    Error_test[k,1], True_Positive_Rate[k,1], False_Positive_Rate[k,1] = errorRate(y_test_est, y_test_np)
    
    
    #Train ANN model with selected hidden units via true positive rate
    n_hidden_units = hidden_opt_tpr[k,0].astype(int)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=1,
                                                       max_iter=max_iter)
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8).data.numpy().squeeze() # threshold output of sigmoidal function
    #y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    #e = (y_test_est != y_test)
    #error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    #Error_test[k,1] = error_rate
    y_test_np = y_test.data.numpy()
    Error_test[k,2], True_Positive_Rate[k,2], False_Positive_Rate[k,2] = errorRate(y_test_est, y_test_np)
#y_test_np = y_test.data.numpy()
#y_test_est_np = y_test_est.data.numpy().squeeze()

#Error_test, True_Positive, False_Positive = errorRate(y_test_est_np, y_test_np)
    
