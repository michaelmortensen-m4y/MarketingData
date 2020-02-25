
from helper import *
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Load marketing data to get attributes names, a pandas dataframe object and a numpy array
attNames, marketingData_pd, marketingData_np = loadMarketingData()
N = len(marketingData_np) # Number of observations

# Only numerical values will be considered for PCA
marketingDataNumericOnly_pd = marketingData_pd[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
marketingDataNumericOnly_np = marketingDataNumericOnly_pd.to_numpy()

# Standardize the data
means = marketingDataNumericOnly_np.mean(axis=0) # get mean of each column
X = marketingDataNumericOnly_np - np.ones((N,1))*means # Get matrix X by substracting the mean from each value in the marketingdata

# PCA by computing SVD of X
U,S,V = svd(X,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Plot variance explained
threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
