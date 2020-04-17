
# We will here attempt to predict continuous values for an attribute using regression techniques such as ordinary least squares regression model

from helper import *
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import matplotlib.pyplot as plt

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
testSize = 0.1 # 10% of the dataset will be taken out at random for testing only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

print("\nComputing linear regression model:")

# Fit linear regression model
#model = lm.LinearRegression()
#model = model.fit(X_train, y_train)
#print(model.score(X_train, y_train))
#print(model.score(X_test, y_test))

# Fit regularized linear ElasticNet regression model with optimal lamda
lamda = 0.0012742749857031334
train_errors = list()
test_errors = list()
print(f"Fitting for lamda={lamda}")
regmodel = lm.ElasticNet(alpha=lamda, max_iter=1000, l1_ratio=0.5)
regmodel = regmodel.fit(X_train, y_train)

# Basemodel as mean
basemodel = y_train.mean()

# ANN
h = 1
print(f"Fitting ANN with h = {h}")
annmodel = nn.MLPRegressor(hidden_layer_sizes=tuple([100]*h))
annmodel = annmodel.fit(X_train, y_train)

# 2-level crossvalidation


