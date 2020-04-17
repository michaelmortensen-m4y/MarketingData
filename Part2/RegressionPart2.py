
# We will here attempt to predict continuous values for an attribute using regression techniques such as ordinary least squares regression model

from helper import *
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

# Load marketing data to get attributes names, a pandas dataframe object and a numpy array
attNames, marketingData_pd, marketingData_np = loadMarketingData()

# Encode categorical attributes as integer for ordinal and 1-of-k for nominal
ordinals = {"education": {
                "illiterate": 1,
                "basic.4y": 2,
                "basic.6y": 3,
                "basic.9y": 4,
                "high.school": 5,
                "professional.course": 6,
                "university.degree": 7,
                "unknown": "WeightedAverage"
    }
}

nominals = [
    "job",
    "marital",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
    "y"
]

attNamesEncoded, marketingDataEncoded_pd, marketingDataEncoded_np = encodeCategorical(marketingData_pd, ordinals=ordinals, nominals=nominals)

# Select attribute to predict (y)
attToPredictName = "education"
attToPredictId = attNamesEncoded.index(attToPredictName)

# Select attributes for the model (X) (all except y for now)
attIdsForModel = list(range(len(attNamesEncoded)))
attIdsForModel.remove(attToPredictId)

# Standardize the data
N, _ = marketingDataEncoded_np.shape
means = marketingDataEncoded_np.mean(axis=0) # get mean of each column
marketingDataEncoded_np_std = marketingDataEncoded_np - np.ones((N,1))*means # Get matrix X by substracting the mean from each value in the marketingdata
marketingDataEncoded_np_std = marketingDataEncoded_np_std*(1/np.std(marketingDataEncoded_np_std,0)) #Deviding by standard deviation to normalize

# Make X and y arrays
X = marketingDataEncoded_np_std[:,attIdsForModel]
y = marketingDataEncoded_np_std[:,attToPredictId]

# Split dataset into training and test set
#testSize = 0.1 # 10% of the dataset will be taken out at random for testing only
X, _, y, _ = train_test_split(X, y, test_size=0.7, random_state=42)

lamdas = np.logspace(-5, 3, num=20)

# 2-level crossvalidation
K = 5
CV_outer = model_selection.KFold(K, shuffle=True, random_state = 42)

k=0
for train_index, test_index in CV_outer.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))
    k += 1

    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    N_par, M_par = X_train.shape
    CV_inner = model_selection.KFold(K, shuffle=True, random_state = 42)

    j=0
    for train_inner, test_inner in CV_inner.split(X_train, y_train):

        # extract training and test set for current CV fold
        print('\nInner Crossvalidation fold: {0}/{1}'.format(j + 1, K))
        X_train_in = X_train[train_inner]
        y_train_in = y_train[train_inner]
        X_test_in = X_train[test_inner]
        y_test_in = y_train[test_inner]
        j += 1

        # Train linear model
        train_errors = list()
        test_errors = list()
        for lamda in lamdas:
            #print(f"Fitting for lamda={lamda}")
            regmodel = lm.ElasticNet(alpha=lamda, max_iter=1000, l1_ratio=0.5)
            regmodel = regmodel.fit(X_train, y_train)
            train_errors.append(regmodel.score(X_train, y_train))
            test_errors.append(regmodel.score(X_test, y_test))

        i_alpha_optim = np.argmax(test_errors)
        alpha_optim = lamdas[i_alpha_optim]
        bestscore = test_errors[i_alpha_optim]

        print(f"Best lamda: {alpha_optim}")
        print(f"Best score: {bestscore}")


        # Train ANN model
        hs = [1, 2, 3, 4, 5]
        train_errors = list()
        test_errors = list()
        for h in hs:
            #print(f"Fitting for h={h}")
            annmodel = nn.MLPRegressor(hidden_layer_sizes=tuple([h]))
            annmodel = annmodel.fit(X_train, y_train)
            train_errors.append(annmodel.score(X_train, y_train))
            test_errors.append(annmodel.score(X_test, y_test))

        i_h_optim = np.argmax(test_errors)
        h_optim = hs[i_h_optim]
        bestscore = test_errors[i_h_optim]

        print(f"Best h: {h_optim}")
        print(f"Best score: {bestscore}")