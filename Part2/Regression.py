
# We will here attempt to predict continuous values for an attribute using regression techniques such as ordinary least squares regression model

from helper import *
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import numpy as np

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
attToPredictId = attNamesEncoded.index("age")

# Select attributes for the model (X) (all except y for now)
attIdsForModel = list(range(len(attNamesEncoded)))
attIdsForModel.remove(attToPredictId)

X = marketingDataEncoded_np[:,attIdsForModel]
y = marketingDataEncoded_np[:,attToPredictId]

# Split dataset into training and test set
testSize = 0.1 # 10% of the dataset will be taken out at random for testing only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

print("\nComputing linear regression model:")

# Fit linear regression model
model = lm.LinearRegression()
model = model.fit(X_train, y_train)

# Try to predict on the test set
y_predicted = model.predict(X_test)

for y_idx, y_p in enumerate(y_predicted):
    print(f"Predicted {y_p} where true value is {y_test[y_idx]}")