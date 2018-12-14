#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 03:19:43 2017

@author: Dish
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

#Importing the shop dataset with pandas
dataset = pd.read_csv('shop_data.csv')
X = dataset.iloc[:, [3, 4]].values

#Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) #ward target minimizing the variance in a cluster
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
#Result shows that 5 clusters are the optimum one

#Fitting hierarchical clustering to the mall dataset
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X) #Now we can see to which cluster each customer is associated

#Visualising the clusters (Will not work in plotting more than 2 dimensions, need to reduce dimensions first)
plt.scatter(X[y_hc == 0, 0], X[y_hc ==0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc ==1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc ==2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc ==3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc ==4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
