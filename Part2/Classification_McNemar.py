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

import matplotlib.pyplot as plt
#import sklearn.linear_model as lm
#imports for decision tree and graphviz plotting
from sklearn import model_selection

from toolbox_02450 import mcnemar


Prediction_ANN_path = 'C:/Users/anjaeh/Desktop/DTU_files/5_Courses/Machine_Learning/Project/Report2/Prediction_ANN.csv'
Prediction_log_path = 'C:/Users/anjaeh/Desktop/DTU_files/5_Courses/Machine_Learning/Project/Report2/Prediction_log.csv'
y_true_path = 'C:/Users/anjaeh/Desktop/DTU_files/5_Courses/Machine_Learning/Project/Report2/y_true_log.csv'
y_true_ANN = 'C:/Users/anjaeh/Desktop/DTU_files/5_Courses/Machine_Learning/Project/Report2/y_true_ANN.csv'

Prediction_ANN = pd.read_csv(Prediction_ANN_path, sep = ',' )
Prediction_log = pd.read_csv(Prediction_log_path)

y_true_log = pd.read_csv(y_true_path)
y_true_ANN = pd.read_csv(y_true_ANN)

print(y_true_log[y_true_log['y_true'] != y_true_ANN['y_true_ANN']])
print(Prediction_log[Prediction_log['baseline'] != Prediction_ANN['y_base_full']])
#gives empty data frame --> y_true from log_reg and ANN script are identical
#the same is true for baseline columns in Prediction_log and Prediction_ANN

#Make one dataframe with all predictions
Prediction_log['ANN'] = Prediction_ANN['y_est_ANN']

#Initalize result matrix for McNemar test
column_names = ['E_theta', 'Confidence Interval', 'p-Value']
mc_nemar = pd.DataFrame(columns = ['E_theta', 'Confidence Interval', 'p-Value'])
result_list = [] 


# Compute the McNemar test confidence level 0.05
alpha = 0.05
#Base vs. Log
result = mcnemar(y_true_log['y_true'], Prediction_log.iloc[:,0], Prediction_log.iloc[:,1], alpha=alpha)
result_list.append(result)
#Base vs. ANN
result = mcnemar(y_true_log['y_true'], Prediction_log.iloc[:,0], Prediction_log.iloc[:,3], alpha=alpha)
result_list.append(result)
#log vs. ANN
result = mcnemar(y_true_log['y_true'], Prediction_log.iloc[:,1], Prediction_log.iloc[:,3], alpha=alpha)
result_list.append(result)
#Base vs. log_balanced
result = mcnemar(y_true_log['y_true'], Prediction_log.iloc[:,0], Prediction_log.iloc[:,2], alpha=alpha)
result_list.append(result)

mc_nemar = pd.DataFrame(result_list, columns = column_names)

out_path = 'C:/Users/anjaeh/Desktop/DTU_files/5_Courses/Machine_Learning/Project/Report2/mcnemar.csv'
mc_nemar.to_csv(out_path, index = False)