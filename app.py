# Importing libraries
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split

# Defining the centroids
centroids = [(-5, -5), (5, 5), (-2.5, 2.5), (2.5, -2.5)]
cluster_std = [1, 1, 1, 1]

# Sample cluser dataset
X, y = make_blobs(n_samples=200, cluster_std=cluster_std,
                  centers=centroids, n_features=2, random_state=2)

# Splitting the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# plt.scatter(X[:, 0], X[:, 1])

# df = pd.read_csv('student_clustering.csv')

# X = df.iloc[:, :].values

# Using KMeans
km = KMeans(n_clusters=4, max_iter=500)
y_means = km.fit_predict(X_train)

# Plotting the points
plt.scatter(X_train[y_means == 0, 0], X_train[y_means == 0, 1], color='red')
plt.scatter(X_train[y_means == 1, 0], X_train[y_means == 1, 1], color='blue')
plt.scatter(X_train[y_means == 2, 0], X_train[y_means == 2, 1], color='green')
plt.scatter(X_train[y_means == 3, 0], X_train[y_means == 3, 1], color='yellow')
plt.show()

# assign clusters to test set
y_means_test = km.assign_clusters(X_test)

# Plotting the points
plt.scatter(X_test[y_means_test == 0, 0], X_test[y_means_test == 0, 1], color='red')
plt.scatter(X_test[y_means_test == 1, 0], X_test[y_means_test == 1, 1], color='blue')
plt.scatter(X_test[y_means_test == 2, 0], X_test[y_means_test == 2, 1], color='green')
plt.scatter(X_test[y_means_test == 3, 0], X_test[y_means_test == 3, 1], color='yellow')
plt.show()