
# This file contains generally useful methods for handling the marketing dataset and more

import os
import sys
import numpy as np
import pandas as pd

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
    print(f"Loaded file {filename} to numpy array with the following attributes and datatypes: ")
    print("\n".join("{:2} {:15} {:5}".format(attNames.index(att), att, dtype) for att, dtype in zip(attNames, dtypes)))

    return attNames, marketingData_df, marketingData_np

