
from helper import *
import matplotlib.pyplot as plt
from itertools import combinations

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
N = len(marketingDataEncoded_np)

# Plot many scatter
attsToUse = [0, 3, 4, 5, 7, 8, 9, 10]
attsNames = ["Age", "Campaign", "Pdays", "Previous", "Con.price.idx", "Con.conf.idx", "Euribor3m", "Nr.employed"]
configs = list(combinations(attsToUse,2))
print(configs)
print(f"Generating {len(configs)} scatterplots...")

for iter, comb in enumerate(configs):
    print(iter+1)
    # Plot scatter with two numeric attributes and a categorial
    xAxisAttribute = comb[0]
    xAttNameIdx = attsToUse.index(xAxisAttribute)
    yAxisAttribute = comb[1]
    yAttNameIdx = attsToUse.index(yAxisAttribute)
    colorAttribute = 11
    numObservationsToPlot = N

    fig = plt.figure()
    xs = marketingDataEncoded_np[:,xAxisAttribute]
    ys = marketingDataEncoded_np[:,yAxisAttribute]
    cs = np.array([ (val.replace('no', 'red').replace('yes', 'blue')) for val in marketingDataEncoded_np[:,colorAttribute]])

    plt.scatter(xs, ys, color=cs, label="red -> y = no \n blue -> y = yes")
    plt.title(f"{attsNames[xAttNameIdx]} vs {attsNames[yAttNameIdx]}")
    plt.xlabel(attsNames[xAttNameIdx])
    plt.ylabel(attsNames[yAttNameIdx])
    plt.legend()

    #plt.show()

# Slow code for the same:
'''
            for p in range(numObservationsToPlot):
                if p % 1000 == 0:
                    print(p)
                thisx = marketingDataEncoded_np[p,xAxisAttribute]
                thisy = marketingDataEncoded_np[p,yAxisAttribute]
                coloridx = None
                #for cats in range(numUniqueValsInColAtt):
                #    if marketingDataEncoded_np[p,numUniqueValsInColAtt+cats] == 1:
                #        coloridx = cats

                if marketingDataEncoded_np[p, colorAttribute] == "no":
                    coloridx = 0
                if marketingDataEncoded_np[p, colorAttribute] == "yes":
                    coloridx = 1

                thiscol = coloridx

                if (thisx != lastx or thisy != lasty or thiscol != lastcol):
                    plt.plot(thisx, thisy, 'o', color=colorsToUse[coloridx])
                    lastx = marketingDataEncoded_np[p,xAxisAttribute]
                    lasty = marketingDataEncoded_np[p,yAxisAttribute]
                    lastcol = coloridx

            plt.legend(["yes", "no"])
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.show()


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
'''

# Plot histogram of age
fig = plt.figure()
plt.hist(marketingData_np[:,0], bins=marketingData_np[:,0].max()-marketingData_np[:,0].min(), align='right')
plt.title("Age histogram")
plt.ylabel("Frequency of calls")
plt.xlabel("Age")

# Plot market values
fig = plt.figure()
plt.plot((marketingDataEncoded_np[:,9]-marketingDataEncoded_np[:,9].mean()) / marketingDataEncoded_np[:,9].std())
plt.plot((marketingDataEncoded_np[:,8]-marketingDataEncoded_np[:,8].mean()) / marketingDataEncoded_np[:,8].std())
plt.plot((marketingDataEncoded_np[:,7]-marketingDataEncoded_np[:,7].mean()) / marketingDataEncoded_np[:,7].std())
plt.plot((marketingDataEncoded_np[:,10]-marketingDataEncoded_np[:,10].mean()) / marketingDataEncoded_np[:,10].std())
plt.title("Social and economic context history")
plt.xlabel("Subsequent calls")
plt.legend(["Euribor3m", "Con.conf.idx", "Con.price.idx", "Nr.employed"])
'''
# Plot histogram of education
educationValues = marketingData_np[:,3]
educationTypes, educationCounts = np.unique(educationValues, return_counts=True)
fig = plt.figure()
plt.bar(educationTypes, educationCounts)

# Plot bar with yes/no/unknown categorical attributes
#fig = plt.figure()
#plt.bar([1, 6, 11], [5, 6, 7], 1, color="red", label="One")
#plt.bar([2, 7, 12], [5, 6, 7], 1, color="blue", label="Two")
#plt.bar([3, 8, 13], [5, 6, 7], 1, color="green", label="Three")

fig = plt.figure()
atts = ['Default','Housing','Loan']
nos = [32588, 18622, 33950]
yess = [3, 21576, 6248]
unknowns = [8597, 990, 990]
def subcategorybar(X, vals, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i],
                width=width/float(n), align="edge")
    plt.xticks(_X, X)
    plt.legend(['No','Yes','Unknown'])
    plt.ylabel("Frequency of calls")
    plt.title("Frequency of yes/no/unknown")

subcategorybar(atts, [nos, yess, unknowns])

#fig = plt.figure()
#jobs = ['admin.','blue-collar','entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
#jobOcc = [10422, 9254, 1456, 1060, 2024, 1720, 1421, 3969, 875, 6743, 1014, 330]
#def subcategorybar(X, vals, width=0.8):
#    n = len(vals)
#    _X = np.arange(len(X))
#    for i in range(n):
#        plt.bar(_X - width/2. + i/float(n)*width, vals[i],
#                width=width/float(n), align="edge")
#    plt.xticks(_X, X, rotation=50)
#    plt.ylabel("Frequency of calls")
#    plt.title("Frequency of job categories")

#subcategorybar(jobs, [jobOcc])
'''
plt.show()
            
