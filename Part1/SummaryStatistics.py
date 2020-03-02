
from helper import *

pd.set_option('display.max_columns', None) # So printing pandas dataframe prints all columns
#pd.set_option('display.max_rows', None)

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
    "poutcome"
]

attNamesEncoded, marketingDataEncoded_pd, marketingDataEncoded_np = encodeCategorical(marketingData_pd, ordinals=ordinals, nominals=nominals)

print("\nComputing summary statistics:")

# Compute and print summary statistics for all attributes
N = len(marketingDataEncoded_pd) # Number of observations
for iter, att in enumerate(attNamesEncoded):
    if att.split("_")[0].lower() not in nominals and "day_of_week" not in att.lower():
        # Print mean and std for originally numeric values
        if att.lower() == "pdays": # Ignore 999 values for the Pdays attribute (only 1515 of 41188 observations have pdays < 999)
            mean = marketingDataEncoded_pd[att][marketingDataEncoded_pd[att] < 999].mean()
            std = marketingDataEncoded_pd[att][marketingDataEncoded_pd[att] < 999].std()
            print(f"{iter:2} {att:25} mean: {mean:.1f} std: {std:.1f}")
        elif att.lower() != "y": # Also ignore y
            mean = marketingDataEncoded_pd[att].mean()
            std = marketingDataEncoded_pd[att].std()
            print(f"{iter:2} {att:25} mean: {mean:.1f} std: {std:.1f}")
    else:
        # Print occurences and 1-of-k mean (share) for originally nominal values
        occ = marketingDataEncoded_pd[att][marketingDataEncoded_pd[att] == 1].size
        share = occ/N
        print(f"{iter:2} {att:25} occ: {occ:5} share: {share:.3f}")
