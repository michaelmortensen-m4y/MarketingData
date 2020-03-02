#Enable python to go into parent folder to open helper.py file
import sys
sys.path.insert(1,'..')

from helper import *
import matplotlib.pyplot as plt

# Load marketing data to get attributes names, a pandas dataframe object and a numpy array
attNames, marketingData_pd, marketingData_np = loadMarketingData()

#print(marketingData_np)

## Visualize correlation between numeric attributes

# Get the datasubset with numeric values
marketingDataNumericOnly_pd = marketingData_pd[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]

# Compute correlation value for each pair of numeric attributes
corr = marketingDataNumericOnly_pd.corr()

# Plot the correlation values in on a heatmap for visualization
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(marketingDataNumericOnly_pd.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(marketingDataNumericOnly_pd.columns)
ax.set_yticklabels(marketingDataNumericOnly_pd.columns)

# Plot the correlation values between categorial attributes using a distance measure
# ...


# Plot histogram of age
fig = plt.figure()
plt.hist(marketingData_np[:,0], bins=marketingData_np[:,0].max()-marketingData_np[:,0].min(), align='right')

# Plot histogram of education
educationValues = marketingData_np[:,3]
educationTypes, educationCounts = np.unique(educationValues, return_counts=True)
fig = plt.figure()
plt.bar(educationTypes, educationCounts)


plt.show()