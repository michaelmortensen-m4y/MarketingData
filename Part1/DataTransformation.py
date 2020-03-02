# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:44:30 2020

@author: anjaeh
"""

#Enable python to go into parent folder to open helper.py file
import sys
sys.path.insert(1,'..')

from helper import *

# Load marketing data to get attributes names, a pandas dataframe object and a numpy array
attNames, marketingData_pd, marketingData_np = loadMarketingData()

#Building outcome vector y and assign classes in dictonary
classLabels = marketingData_np[:,-1] 
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))

# Extract outcome vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

#Drop outcome variable from data
marketingData_np = np.delete(marketingData_np, 20, axis = 1)

#Transformation of categorical attributes with 1 out of K encoding
#Building a list of dictonaries for accessing the variables

cat = [1,2,4,5,6,7,8,9,14]
cat_attNames = []
cat_attributes_encoded = []
marketingData_np_int = marketingData_np.copy()
marketingData_np_outofK = np.delete(marketingData_np, cat, axis = 1)
for i in cat :
    Label = marketingData_np[:,i]
    Names = np.unique(Label)
    Dictionary = dict(zip(Names,range(len(Names))))
    cat_attNames.append(Dictionary)
    #Modifying data table to replace strings with integers
    vector = np.asarray([Dictionary[value] for value in Label])
    marketingData_np_int[:,i] = vector
    
    #1 out of K encoding of variables
    vector_T = vector.T
    K = vector.max()+1    
    encoding = np.zeros((vector.size, K))
    encoding[np.arange(vector.size), vector] = 1
    
    #Forming a list of 1 out of K encoded arrays (not really necessary)
    #Appending data table with out of K encoded attributes --> order of attributes is different now
    cat_attributes_encoded.append(encoding)
    marketingData_np_outofK = np.concatenate((marketingData_np_outofK, encoding), axis = 1)


#Encoding of Education variable as ordinal
marketingData_Edu = marketingData_np[:,3]
EduDict = {'illiterate':0, 'basic.4y':1, 'basic.6y':2,'basic.9y':3, 'high.school':4, 'professional.course':5, 'university.degree':6 }
marketingData_Edu_drop = marketingData_Edu[marketingData_Edu != 'unknown']
Edu_vector = np.asarray([EduDict[value] for value in marketingData_Edu_drop])

#Calculating mean and imputing value for all unknown variables in education
Key_unknown = round(Edu_vector.mean(),1)
EduDict.update(unknown = Key_unknown)

#Creating full vector of education values and updating data table
Edu_vector = np.asarray([EduDict[value] for value in marketingData_Edu])
marketingData_np_int[:,3] = Edu_vector
marketingData_np_outofK[:,1] = Edu_vector




