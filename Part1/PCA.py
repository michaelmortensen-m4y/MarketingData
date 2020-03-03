#Enable python to go into parent folder to open helper.py file
import sys
sys.path.insert(1,'..')

from helper import *
from SummaryStatistics import ordinals, nominals
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Load marketing data to get attributes names, a pandas dataframe object and a numpy array
attNames, marketingData_pd, marketingData_np = loadMarketingData()
attNamesEncoded, marketingDataEncoded_pd, marketingDataEncoded_np = encodeCategorical(marketingData_pd, ordinals=ordinals, nominals=nominals)
N = len(marketingData_np) # Number of observations

#Extract outcome vector y, convert to np
classLabels = marketingData_np[:,-1] 
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.asarray([classDict[value] for value in classLabels])

# Compute values of C --> number of classes
C = len(classNames)

############## Only Numerical Attributes ###############

# Only numerical values will be considered for PCA
marketingDataNumericOnly_pd = marketingData_pd[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
marketingDataNumericOnly_np = marketingDataNumericOnly_pd.to_numpy()

attNames_num = list(marketingDataNumericOnly_pd.dtypes.keys())

# Standardize the data
means = marketingDataNumericOnly_np.mean(axis=0) # get mean of each column
X_num = marketingDataNumericOnly_np - np.ones((N,1))*means # Get matrix X by substracting the mean from each value in the marketingdata
X_num = X_num*(1/np.std(X_num,0)) #Deviding by standard deviation to normalize

# PCA by computing SVD of X
U,S,V_num = svd(X_num,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Plot variance explained
threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components - Only numerical attributes');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#Plotting data on PC1 and PC2 space according to outcome class
#Transpose of V
V = V_num.T    

# Project the centered data onto principal component space
Z = X_num @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title('Marketing Data - Numerical: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()

#### Attribute coefficients in principal component space for Numerical Attributes ####
for att in range(V.shape[1]):
    plt.arrow(0,0, V[att,i], V[att,j])
    plt.text(V[att,i], V[att,j], attNames_num[att])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.grid()
    # Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title('Marketing data - Numerical attributes')
plt.axis('equal')
    
plt.show()

############### FULL Dataset #################

#full dataset with 1-out-of-K encoded categorial variables
marketingDataEncoded_full_pd = marketingDataEncoded_pd.drop('y', axis = 1)
marketingDataEncoded_full_np = marketingDataEncoded_full_pd.to_numpy()

# Standardize the data
means = marketingDataEncoded_full_np.mean(axis=0) # get mean of each column
X_full = marketingDataEncoded_full_np - np.ones((N,1))*means # Get matrix X by substracting the mean from each value in the marketingdata
X_full = X_full*(1/np.std(X_full,0)) #Deviding by standard deviation to normalize

# PCA by computing SVD of X
U,S,V_full = svd(X_full,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Plot variance explained
threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components - Full dataset');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#Plotting data on PC1 and PC2 space according to outcome class
#Transpose of V
V = V_full.T    

# Project the centered data onto principal component space
Z = X_full @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title('Marketing Data - Full: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()




