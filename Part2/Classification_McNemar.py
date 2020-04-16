# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:24:51 2020

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

from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary, mcnemar
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


#Make prediction with selected models

#regression model:


#ANN:
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



# Compute the McNemar test confidence level 0.05
alpha = 0.05
[theta_pred, CI, p] = mcnemar(y_true, y_test_est[:,0], y_test_est[:,3], alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)