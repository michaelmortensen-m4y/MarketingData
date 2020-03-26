# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:20:56 2020

@author: anjaeh
"""

from os import getcwd
import pandas as pd
#Enable python to go into parent folder to open helper.py file
import sys
sys.path.insert(1,'..')

from helper import *
from Part1.SummaryStatistics import ordinals, nominals
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
#imports for decision tree and graphviz plotting
from sklearn import tree
from platform import system
from toolbox_02450 import windows_graphviz_call
from matplotlib.image import imread



# Load marketing data to get attributes names, a pandas dataframe object and a numpy array
attNames, marketingData_pd, marketingData_np = loadMarketingData()
attNamesEncoded, marketingDataEncoded_pd, marketingDataEncoded_np = encodeCategorical(marketingData_pd, ordinals=ordinals, nominals=nominals)
N = len(marketingData_np) # Number of observations

#Extract outcome vector y, convert to np
classLabels = marketingData_np[:,-1] 
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.asarray([classDict[value] for value in classLabels])

# Compute values of C --> number of classes
C = len(classNames)

#full dataset with 1-out-of-K encoded categorial variables
marketingDataEncoded_full_pd = marketingDataEncoded_pd.drop('y', axis = 1)
marketingDataEncoded_full_np = marketingDataEncoded_full_pd.to_numpy()

attNamesEncoded.remove('y')

# Standardize the data
means = marketingDataEncoded_full_np.mean(axis=0) # get mean of each column
X_full = marketingDataEncoded_full_np - np.ones((N,1))*means # Get matrix X by substracting the mean from each value in the marketingdata
X_full = X_full*(1/np.std(X_full,0)) #Deviding by standard deviation to normalize

#Logistic regression on whole dataset
model = lm.LogisticRegression()
model = model.fit(X_full,y)

#Predict outcome for training data set --> just for playing around!
y_est = model.predict(X_full)
y_prob = model.predict_proba(X_full)

misclass_rate = np.sum(y_est != y) / float(len(y_est))

f = plt.figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plt.plot(class0_ids, y_prob[class0_ids], ',y')
class1_ids = np.nonzero(y==1)[0].tolist()
plt.plot(class1_ids, y_prob[class1_ids], ',r')
plt.xlabel('Customer'); plt.ylabel('Predicted prob. of "no"');
plt.legend(['No', 'Yes'])
plt.ylim(-0.01,1.5)

plt.show()

#Testing a tree on the whole dataset
# Fit regression tree classifier, Gini split criterion, no pruning
criterion='gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=50, max_depth = 4)
dtc = dtc.fit(X_full,y)

#Plotting the tree
fname='tree_' + criterion
# Export tree graph .gvz file to parse to graphviz
out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attNamesEncoded)

# Depending on the platform, we handle the file differently, first for Linux 
# Mac
if system() == 'Linux' or system() == 'Darwin':
    import graphviz
    # Make a graphviz object from the file
    src=graphviz.Source.from_file(fname + '.gvz')
    print('\n\n\n To view the tree, write "src" in the command prompt \n\n\n')
    
# ... and then for Windows:
if system() == 'Windows':
    # N.B.: you have to update the path_to_graphviz to reflect the position you 
    # unzipped the software in!
    windows_graphviz_call(fname=fname,
                          cur_dir=getcwd(),
                          path_to_graphviz=r'C:\Users\anjaeh\Desktop\DTU_files\5_Courses\Machine_Learning\02450Toolbox_Python\graphviz-2.38\release')
    plt.figure(figsize=(12,12))
    plt.imshow(imread(fname + '.png'))
    plt.box('off'); plt.axis('off')
    plt.show()