X= dlmread ('mfcc.csv',',');
mu = mean(X)
Xm = bsxfun(@minus, X, mu);
C = cov(Xm)
[V,D] = eig(C)
[D, i] = sort(diag(D), 'descend');
V = V(:,i);
cumsum(D) / sum(D)