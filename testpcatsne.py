import numpy as np
from numpy import genfromtxt

#from sklearn.datasets import fetch_mldata

#mnist = fetch_mldata("MNIST original")
#X = mnist.data / 255.0
#y = mnist.target

X = genfromtxt('mfcc.csv', delimiter=',',dtype=None)
y = genfromtxt('classes.csv', delimiter=',',dtype=None)

print (X.shape, y.shape)

import pandas as pd

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i)) #cast to string

X, y = None, None

#print ('Size of the dataframe: {}'.format(df.shape))
'''
rndperm = np.random.permutation(df.shape[0])

#import matplotlib.pyplot as plt

# Plot the graph
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,30):
    ax = fig.add_subplot(3,10,i+1, title='Digit: ' + str(df.loc[rndperm[i],'label']) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
plt.show()
'''
################# PCA #################33
from sklearn.decomposition import PCA

pca = PCA(n_components=3) # With 3 principal components because we can see three-dimensional graphs
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]

print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)) # how much of the variation they actually account for

'''
from ggplot import *

chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
print (chart)'''