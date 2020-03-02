
from helper import *
import matplotlib.pyplot as plt

# Load marketing data to get attributes names, a pandas dataframe object and a numpy array
attNames, marketingData_pd, marketingData_np = loadMarketingData()


#print(marketingData_np)

## Visualize correlation between numeric attributes

# Get the datasubset with numeric values
#marketingDataNumericOnly_pd = marketingData_pd[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]

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
    "poutcome"
]

attNamesEncoded, marketingDataEncoded_pd, marketingDataEncoded_np = encodeCategorical(marketingData_pd, ordinals=ordinals, nominals=nominals)

# Plot scatter with two numeric attributes and a categorial
fig = plt.figure()
colorsToUse = ["red", "blue", "green", "brown", "orange", "black", "pink", "purple", "yellow", "black", "black", "black"]
colors = []
for p in range(1000):
    coloridx = int(marketingDataEncoded_np[p,1])
    if coloridx in [1, 2, 3]:
        plt.plot(marketingDataEncoded_np[p,0], marketingDataEncoded_np[p,2], 'o', alpha=.5, color=colorsToUse[coloridx])

# Compute correlation value for each pair of numeric attributes
corr = marketingDataEncoded_pd.corr()

# Plot the correlation values in on a heatmap for visualization
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(marketingDataEncoded_pd.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(marketingDataEncoded_pd.columns)
ax.set_yticklabels(marketingDataEncoded_pd.columns)


# Plot histogram of age
fig = plt.figure()
plt.hist(marketingData_np[:,0], bins=marketingData_np[:,0].max()-marketingData_np[:,0].min(), align='right')

# Plot histogram of education
educationValues = marketingData_np[:,3]
educationTypes, educationCounts = np.unique(educationValues, return_counts=True)
fig = plt.figure()
plt.bar(educationTypes, educationCounts)


plt.show()