
# This file contains generally useful methods for handling the marketing dataset and more

import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy

def loadMarketingData(csvFilename='bank-additional-full.csv'):

    # Get filepath
    filename = csvFilename
    if filename[-4:] != '.csv':
        filename = filename + '.csv'

    this_dir = os.path.abspath(os.path.dirname(__file__))
    dataFilepath = os.path.join(this_dir, 'MarketingData', filename)

    # Read file to numpy array using pandas
    marketingData_df = pd.read_csv(dataFilepath, sep=';')
    marketingData_np = marketingData_df.to_numpy()

    # Get infered attribute names and datatypes
    attNames = list(marketingData_df.dtypes.keys())
    dtypes = []
    for col in marketingData_np[0]:
        dtypes.append(str(type(col)).replace("<class '", "").replace("'>", ""))

    # Print status
    print(f"Loaded file {filename} to pandas dataframe and numpy array with the following attributes and datatypes: ")
    print("\n".join("{:2} {:15} {:5}".format(attNames.index(att), att, dtype) for att, dtype in zip(attNames, dtypes)))

    return attNames, marketingData_df, marketingData_np


def encodeCategorical(marketingData_pd, ordinals=None, nominals=None):

    # Integer encode ordinals with the values in ordinals dict to a temporary dataframe needed for computing weighted average
    marketingData_pd_t = marketingData_pd.replace(ordinals, inplace=False)

    # Replace strings ("average", ...) with the corresponding value for each attribute
    for attname, subdic in ordinals.items():
        for subdickey, subdicValue in subdic.items():
            if isinstance(subdicValue, str):
                if subdicValue.lower() == "weightedaverage":
                    weightedaverage = marketingData_pd_t[pd.to_numeric(marketingData_pd_t[attname], errors="coerce").notnull()][attname].mean()
                    ordinals[attname][subdickey] = weightedaverage
                    print(f"Inserted weighted average = {weightedaverage} for category {subdickey} for attribute {attname}")

    # Integer encode ordinals with the final values given in the dict parameter ordinals where strings have been replaced
    marketingData_pd.replace(ordinals, inplace=True)

    # 1-of-k encode the nominals given in the list parameter nominals
    marketingData_pd = pd.get_dummies(marketingData_pd, columns=nominals)

    # Get numpy array
    marketingData_np = marketingData_pd.to_numpy()

    # Get new attribute names and datatypes
    attNamesEncoded = list(marketingData_pd.dtypes.keys())
    dtypes = []
    for col in marketingData_np[0]:
        dtypes.append(str(type(col)).replace("<class '", "").replace("'>", ""))

    # Print status
    print("Encoded dataset to pandas dataframe and numpy array with the following attributes and datatypes: ")
    print("\n".join("{:2} {:15} {:5}".format(attNamesEncoded.index(att), att, dtype) for att, dtype in zip(attNamesEncoded, dtypes)))

    return attNamesEncoded, marketingData_pd, marketingData_np

