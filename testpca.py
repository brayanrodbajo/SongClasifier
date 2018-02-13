import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA

X_from_data = genfromtxt('mfcc1.csv', delimiter=',',dtype=None)
print(X_from_data)
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X_from_data)


#1PCA
print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  