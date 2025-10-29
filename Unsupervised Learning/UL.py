# %%

# Introduction to Unsupervised Learning
# Unsupervised machine learning refers to the category of machine learning techniques
# where models are trained on a dataset without labels. Unsupervised learning is generally
# use to discover patterns in data and reduce high-dimensional data to fewer dimensions.

# Clustering 
# Clustering is the process of grouping objects from a dataset such that objects in the same group
# (called a cluster) are more similar (in some sense) to each other than to those in other
# groups. https://scikit-learn.org/stable/modules/clustering.html

# Real world Applications of Clustering
#1 Customer segmentation
#2 Product Recommendation 
#3 Feature engineering
#4 Anamoly/fraud detection
#5 Taxonomy Creation

# We'll use the Iris Flower dataset to study some of the clustering algorithms available in
# scikit-learn. It contains various measurements for 150 flowers belonging to 3 different species

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

iris_df = sns.load_dataset('iris')

# %%
iris_df

# %%
sns.scatterplot(data=iris_df, x='sepal_length', y='petal_length', hue='species')

# %%
# We'll attempt to cluster observations using numeric columns in the data
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = iris_df[numeric_cols]
X

# %%
# K Means Clustering
# The K-Means algorithm attempts to classify objects into a pre-determined number of clusters by
# finding optimal central points (called centroids) for each cluster. Each object is classified
# as belonging to the cluster repressented by the closest centroid

# Here's how the K-means algorithm works:
#1 Pick K random objects as the initial cluster centers. 
#2 Classify each object into the cluster whose center is closest to the point.
#3 For each cluster of classified objects, compute the centroid (mean).
#4 Now reclassify each object using the centroids as cluster centers.
#5 Calculate the total variance of the clusters (this is the measure of goodness)
#6 Repeat steps 1 to 6 a few more times and pick the cluster centers with the lowest total variance

# %%
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

# %%
preds = model.predict(X)
preds

# %%
sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=preds)
centers_x, centers_y = model.cluster_centers_[:,0], model.cluster_centers_[:,2]
plt.plot(centers_x, centers_y, 'xb')

# %%
# As you can see, K-means algorithm was able to classify (for the most part) different specifies
# of flowers into seperate clusters. Note that we did not provide the "species" column as an input
# to KMeans.

# We can check the "goodness" of the fit by looking at model.inertia_, which contains the sum
# of squared distances of samples to their closest cluster center. Lower the inertia, better the fit.

model.inertia_

# %%
model = KMeans(n_clusters=6, random_state=42).fit(X)

# %%
preds = model.predict(X)
preds

# %%
sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=preds)

# %%
model.inertia_

# %%
# In most real-world scenarios, there's no predetermined number of clusters. In such a case, you
# can create a plot of "No. of clusters" vs "Inertia" to pick the right number of clusters.

options = range(2,11)
inertias = []

for n_clusters in options:
    model = KMeans(n_clusters, random_state=42).fit(X)
    inertias.append(model.inertia_)

plt.title("No. of clusters vs. Inertia")
plt.plot(options, inertias, '-o')
plt.xlabel('No. of clusters (K)')
plt.ylabel('Inertia')


# %%
# Mini Batch K Means: The K-means algorithm can be quite slow for really large dataset. Mini-batch
# K-means is an iterative alternative to K-means that works well for large datasets. Learn more
# about it here: https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans

# Exercise: Perform clustering on the Mall customers dataset on Kaggle. Study the segments 
# carefully and report your observations. 

# %%
# DBScan
# Density-based spatial clustering of applications with noise(DBSCAN) uses the density of points
# in a region to form clusters. It has two main parameters: "epsilon" and  "min samples" using 
# which it classifies each point as a core point, reachable point or noise point(outlier)

from sklearn.cluster import DBSCAN

# %%
model = DBSCAN(eps=1.1, min_samples=4)
model.fit(X)

# %%
# In DBSCAB, there's no prediction step. It directly assigns labels to all inputs.

model.labels_

# %%
sns.scatterplot(data=X, y='petal_length', x='sepal_length', hue=model.labels_)
# %%
# Exercise: Try changing the values of eps and min_samples and observe how the number of clusters
# the classification changes

model = DBSCAN(eps=0.41, min_samples=3).fit(X)
sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=model.labels_)

# DBSCAN utilizes nearness of points whereas k-means utilizes distance based clustering.
# %%
# Hierarchical Clustering
# Creates a hierarchy or a tree of clusters.

# While there are several approaches to hierarchical clustering, the most common approach works
# as follows:

#1 Mark each point in the dataset as a cluster.
#2 Pick the two closest cluster centers wihtout a parent and combine them into a new cluster.
#3 The new cluster is the parent cluster of the two clusters, and its center is the mean of
#  all the points in the cluster.
#4 Repeat steps 2 and 3 till there's just one cluster left.

#Exercise: Implement Hierarchical clustering for the Iris dataset using scikit-learn.

# %%
# Dimensional Reduction and Manifold Learning

# In machine learning problems, we often encounter datasets with a very large number of dimensions
# (features or columns). Dimensionality reduction techniques are used to  reduce the number of 
# dimensions or features with the data to a manageable or convenient number.

# Applications of dimensionality reduction:
#1 Reducing size and data without loss of information
#2 Training machine learning models efficiently
#3 Visualizing high-dimensional data in 2/3 dimensions

# %%
# Principal Component Analysis (PCA)

# Principal Component is a dimensionality reduction technique that uses linear projections of
# data to reduce their dimensions, while attempting to maximise the variance of data in the
# projection.
# Watch this video to learn how PCA works: https://www.youtube.com/watch?v=FgakZw6K1QQ

# Let's apply PCA to the Iris dataset.

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(iris_df[numeric_cols])

# %%
transformed = pca.transform(iris_df[numeric_cols])
sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=iris_df['species'])

# %%
# As you can see, the PCA algorithm has done a very good job of seperating different species of
# flowers using just 2 measures.

#Exercise: Apply Principal Component Analysis to a large high-dimensional dataset and train a machine
# learning model using the low-dimensional results. Observe the changes in the loss and training 
# time for different numbers of target dimensions.

# Learn more here: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

# %%
# t-Distributed Stochastic Neighbor Embedding (t-SNE)

# Manifold learning is an approach to non-linear dimensionality reduction. Algorithms for this 
# task are based on the idea  that the dimensionality of many data sets is only artificially high.
# Scikit-learn provides many algorithms for manifold learning: https://scikit-learn.org/stable/modules/manifold.html
# A commonly-used manifold learning technique is t-Distributed Stochastic Neighbor Embedding or
# t-SNE, used to visualize high dimensional data in one, two or three dimensions.

from sklearn.manifold import TSNE

# %%
tsne = TSNE(n_components=2)
transformed = tsne.fit_transform(iris_df[numeric_cols])
sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=iris_df['species'])

# %%
# PCA is good for machine learning but t-SNE is good for visualizing purpose.
# Exercise: Use t-SNE to visualize the MNIST handwritten digits dataset.

# %%
