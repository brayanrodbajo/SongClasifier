% http://archive.ics.uci.edu/ml/datasets/Wine
data = dlmread("wine.data",",");
y = data(:,1);
X = data(:,2:end);
pca(X)
