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
    
    y_est = np.asarray(y_est)
    y_true = np.asarray(y_true)
    
    False_positive = float(len(y_est[np.logical_and(y_est == 1, y_true == 0)] ))
    False_negative = float(len(y_est[np.logical_and(y_est == 0, y_true == 1)]))
    true_positive = float(len(y_est[np.logical_and(y_est == 1, y_true == 1)]))
    true_negative = float(len(y_est[np.logical_and(y_est == 0, y_true == 0)]))
    true_positive_rate = true_positive/(true_positive + False_negative)
    false_positive_rate = False_positive/(False_positive + true_negative)
    
    return error_test, true_positive_rate, false_positive_rate