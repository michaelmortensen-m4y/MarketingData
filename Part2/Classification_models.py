# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:22:35 2020

@author: anjaeh
"""
import numpy as np
#import pandas as pd

#set method returns unique values in the iterable
#key customizes sorting sorting of the variable 
def baseline_classification(train, test):
    #y_train = [row[-1] for row in train]
    y_train = train.tolist()
    classes = set(y_train)
    base_prediction = max(classes, key=y_train.count)
    y_est = [base_prediction for i in range(len(test))]
    return y_est



def errorRate(y_est, y_true):
    """calculates misclassification rate"""
    error_test = np.sum(y_est != y_true) / float(len(y_true))
    
    return error_test