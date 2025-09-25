
# Vinho Verde Quality K Means Clustering

Find the dataset [Here](https://www.kaggle.com/datasets/ruthgn/wine-quality-data-set-red-white-wine?resource=download)

## Overview

Vinho Verde, which translates to “green wine,” hails from the North of Portugal. It’s a crisp,
aromatic, and low-alcohol wine, often with a slight spritz. Most of the wines classified as
Vinho Verde are white, but the region is also known to produce red and rosé wines. The white
Vinho Verde is very fresh, due to its natural acidity, with fruity and floral aromas that depend
on the grape variety.

The red and rosé Vinho Verde wines are much less common than the white ones. That is
caused mainly by the region's climatic conditions with its relatively cool temperatures and
high level of rainfall that make it impossible for the red wine grapes to ripen.

K-Means Clustering is an unsupervised learning algorithm that aims to group data based on
their similarity. In the context of wine quality prediction, K- Means clustering can be a
powerful tool for discovering patterns and segmenting wines into distinct clusters. K-means
clustering can identify natural groupings of wines with similar characteristics. By clustering
wines into distinct groups, we gain insights into their inherent qualities and patterns. For
instance, wines within the same cluster might share common flavor profiles or production
methods. Additionally, K-means clustering can aid in feature selection for subsequent
predictive models, helping to identify which attributes significantly impact wine quality.

## Objective 
The objective of using K-means clustering to predict the quality of Vinho Verde wine is to
discover natural groupings within the wine dataset. K-Means aims to divide the wines into
clusters based on their similarity. By doing so, it identifies groups of wines that share common
characteristics, such as flavor profiles, chemical composition, or production methods.

### Questions to Answer:

Which Vinho Verde wines are like each other?

Which features significantly impact the grouping of Vinho Verde wines?

Can we gain insights into the inherent qualities of Vinho Verde wines from the observed
patterns in the clusters?

## Analysis

### Exploratory Analysis

This data set contains records related to red and white variants of the Portuguese Vinho Verde
wine. It contains information from 1599 red wine samples and 4898 white wine samples. Input
variables in the data set consist of the type of wine (either red or white wine) and metrics from
objective tests (e.g. acidity levels, PH values, ABV, etc.), while the target/output variable is a
numerical score based on sensory data.

Attribute Information:

    type of wine: type of wine (categorical: 'red', 'white')
    
    fixed acidity: The acids that naturally occur in the grapes are used to ferment the wine
    and carry over into the wine. They mostly consist of tartaric, malic, citric, or succinic
    acid that mostly originate from the grapes used to ferment wine. They also do not
    evaporate easily. (g / dm^3)
    
    volatile acidity: Acids that evaporate at low temperatures—mainly acetic acid which
    can lead to an unpleasant, vinegar-like taste at very high levels. (g / dm^3)
    citric acid: Citric acid is used as an acid supplement which boosts the acidity of wine.
    It's typically found in small quantities and can add 'freshness' and flavor to wines. (g /
    dm^3)
    
    residual sugar: The amount of sugar remaining after fermentation stops. It's rare to find
    wines with less than 1 gram/liter. Wines residual sugar level greater than 45 grams/liter
    are considered sweet. On the other end of the spectrum, wine that does not taste sweet
    is considered dry. (g / dm^3)
    
    chlorides: The amount of chloride salts (sodium chloride) present in the wine. (g /
    dm^
    
    free sulfur dioxide: The free form of SO2 exists in equilibrium between molecular SO
    (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of
    wine. All else constant, the higher the free sulfur dioxide content, the stronger the
    preservative effect. (mg / dm^3)
    
    total sulfur dioxide: The amount of free and bound forms of S02; in low
    concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations
    over 50 ppm, SO2 becomes evident in the nose and taste of wine. (mg / dm^3)
    
    density: The density of wine juice depending on the percent alcohol and sugar content;
    it's typically similar but higher than that of water (wine is 'thicker'). (g / cm^3)

    pH: A measure of the acidity of wine; most wines are between 3-4 on the pH scale. The
    lower the pH, the more acidic the wine is the higher the pH, the less acidic the wine.
    (The pH scale technically is a logarithmic scale that measures the concentration of free
    hydrogen ions floating around in your wine. Each point of the pH scale is a factor of 10.
    This means a wine with a pH of 3 is 10 times more acidic than a wine with a pH of 4)
        
    sulphates: Amount of potassium sulphate as a wine additive which can contribute to
    sulfur dioxide gas (S02) levels; it acts as an antimicrobial and antioxidant agent.(g /
    dm3)
    
    alcohol: How much alcohol is contained in each volume of wine (ABV). Wine
    generally contains between 5–15% of alcohols. (% by volume)

Output variable:

    quality: score between 0 (very bad) and 10 (very excellent) by wine experts

 <img width="893" height="796" alt="image" src="https://github.com/user-attachments/assets/9398d374-a4db-4547-9d7b-2c60346a566d" />

Figure 1. Correlation Map

### Some observations from the correlation map and exploratory analysis are:

    This correlation map shows that Alcohol has a positive correlation with quality,
    suggesting that higher alcohol content tends to be associated with better wine quality.
    Wines with higher sulphate levels are associated with better quality. Sulphates may
    enhance wine preservation and flavor.
    
    There is a positive correlation between total sulfur dioxide and free sulfur dioxide. As
    the total sulfur dioxide content increases, the free sulfur dioxide content tends to
    increase as well.
    
    Interestingly, there is a positive correlation between residual sugar and quality. Sweeter
    wines tend to receive higher quality ratings.

    There is a negative correlation between volatile acidity and quality. Lower volatile
    acidity is associated with higher wine quality. High volatile acidity can lead to
    unpleasant flavors.
    
    The chloride content in wine shows a negative correlation with quality. Lower chloride
    levels are preferable for better-tasting wines.

## Preprocessing 
Before applying K-means, it is essential to prepare the data to ensure accurate and meaningful
results.

### Data Cleaning 
Clean data ensures accurate model training. Preprocessing helps handle
missing values, outliers, and inconsistencies. K-means is sensitive to noisy data, so removing
or inputting missing values is essential. This data set did not contain any missing values and
did not require additional cleaning.

### Data Transformation
K-means relies on distance-based measurements. K-Means algorithm
cannot use categorical features even if you convert them to numeric features. The wine type
categorial column was dropped from the data frame to ensure the data only includes numerical
data. The data was then normalized using standard scaler to reduce the impact of outliers and
noise, making it easier for the algorithm to identify clusters.

## Cluster Development

### Setting up K-Means

The clusters were developed using KMeans from sklearn.cluster

```
clusterNum = 3  

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
```

The variable clusterNum represents the desired number of clusters you want to create using the
K-means algorithm. Next, we initialize the K-means clustering algorithm.

    KMeans : Refers to the K-means clustering model.
    init = "k-means++": Specifies the method for initializing the cluster centroids. "k-

means++" is a smart initialization technique that helps improve convergence speed and
quality of the final clusters.

    n_clusters = clusterNum: Sets the number of clusters to the value stored in the
    clusterNum variable (which is 3 in this case).
    n_init = 12: Determines the number of times the K-means algorithm will be run with
    different initializations. The result will be the best one out of these runs based on the
    lowest inertia.

## Fitting the Model

k_means.fit(X)

<img width="321" height="75" alt="image" src="https://github.com/user-attachments/assets/ce02fb76-bc33-40e1-8acf-c192cde2b46c" />

Figure 2. K-Means Model

This line of code trains the K-means model using the provided data X. The algorithm will find
the cluster centroids and assign each data point to one of the clusterNum clusters based on
their similarity.

In summary, this code initializes the K-means clustering algorithm with 3 clusters, using smart
centroid initialization ("k-means++"), running it 12 times to find the best clustering solution,
and fits the model to the data stored in the variable X.

## Results

### Cluster Properties

Silhouette Coefficient(silhouette_coef = silhouette_score(X, labels )): The Silhouette
Coefficient ranges from -1 (poor clustering) to 1 (well-separated clusters), with values near 0
indicating overlapping clusters. A Silhouette Coefficient of 0.23 is relatively low, suggesting
that the clusters might be overlapping, and the data points are not distinctly separated into their
respective clusters. This could imply that the wine quality features do not form distinct
groupings when partitioned into clusters using K-Means, or it could be an indication that a
different number of clusters might yield better results.

Scatter Plots:

<img width="574" height="442" alt="image" src="https://github.com/user-attachments/assets/8628bc5e-6077-4992-b020-5c90670bbccd" />

Figure 3. Volatile Acidity, Alcohol Content, and pH Scatterplot

The scatter plot visualizes the relationship between three wine characteristics: volatile acidity,
alcohol content, and pH. Wines with similar volatile acidity, alcohol content, and pH tend to
cluster together .The color of the data points indicate the pH level. The red dots represent the
cluster centers obtained after applying k-means clustering to the wine data. Each red dot
corresponds to the center of one of these clusters. These centers represent typical
characteristics for each group of wines. Wines with higher volatile acidity tend to have lower
alcohol content. Conversely, wines with lower volatile acidity have higher alcohol content.

<img width="1027" height="1027" alt="image" src="https://github.com/user-attachments/assets/f7784c49-7859-4f83-aff5-2d010860805a" />

Figure 4. D scatter plot representing volatile acidity, alcohol content, and Free Sulphur
Dioxide

This scatter plot visualizes the relationship between three wine characteristics: volatile acidity,
free sulfur dioxide, and alcohol content. The data points are clustered using K-means
clustering to identify patterns or groups within the data. Each color represents a different
cluster, indicating wines with similar chemical compositions.

## Output Interpretation

The objective of using K-means clustering to predict the quality of Vinho Verde wine is to
discover natural groupings within the wine dataset. This objective has been partially met. The
high inertia suggests that the data points within clusters are spread out. We can aim for a lower
inertia by refining the cluster centroids or exploring different K values. The positive silhouette
coefficient indicates some separation between clusters. A higher coefficient would imply
better-defined clusters. We can explore feature engineering to improve this. The intra distances
show that data points within clusters are relatively close. While reasonable, we can strive for
even tighter cohesion within clusters.

**Evaluation** : employ appropriate metrics (measures) to quantitatively evaluate the performance

of the clusters. For unsupervised classification, this primarily involves distance metrics.

**Intra-cluster Distance** : Intra-cluster distance measures the average distance between data
points within the same cluster. Lower values indicate tighter groupings, while higher values
suggest greater variability.

**Cluster 0**: The average distance between data points within this cluster is
approximately 2.38. This indicates that the data points within Cluster 0 are relatively
close to each other in terms of their wine quality features.

**Cluster 1** : The intra-cluster distance for Cluster 1 is approximately 2.33. Similar to
**Cluster 0**, this suggests that the data points within Cluster 1 are also relatively tightly
grouped based on their wine quality characteristics.
**Cluster 2** : The highest intra-cluster distance is observed in Cluster 2, with a value of
approximately 2.85. This implies that the data points within Cluster 2 are more spread
out compared to the other clusters, indicating greater variability in wine quality
features.

**Inter-cluster Distance**: Inter-cluster distance measures the average distance between
centroids of different clusters. It reflects how well-separated the clusters are. The lower the
inter-cluster distance (inertia), the better the separation between clusters. The inertia value of
46557.99 is high. A high inertia suggests that the clusters are not well-defined. It implies that
the data points within each cluster are less cohesive and have a greater variance. This high
inertia might indicate that the chosen number of clusters is not optimal or that the features used
for clustering are not effectively capturing the underlying patterns.

**Centroids(k_means.cluster_centers_)**: Each centroid represents a group of wines that have
similar physicochemical properties. The values of the centroids are the average values of the
wines in each cluster for the properties. The first centroid has these values:

    Fixed Acidity (1.17362995): High, possibly due to grape type or winemaking process.
    
    Volatile Acidity (-0.37923873): Low, suggesting fewer unwanted flavors.
    
    Citric Acid (-0.59470841): Low, indicating less tartness.
    
    Residual Sugar (0.90228927): High, suggesting a sweeter taste.
    
    Chlorides (-0.82605727): Low, indicating less saltiness.
    
    Free Sulfur Dioxide (-1.1330634): Low, suggesting less oxidation and spoilage.
    
    Total Sulfur Dioxide (0.65609982): High, possibly due to winemaking process.
    
    Density (0.54814444): High, possibly due to sugar and alcohol content.
    
    pH (0.78038037): High, indicating less acidity.
    
    Sulphates (-0.16943267): Low, suggesting less likelihood of causing wine allergies.
    
    Alcohol (-0.34709679): Low, possibly due to grape variety or fermentation process.

These values suggest that the wines in this cluster have relatively high fixed acidity and
residual sugar, and low volatile acidity and chlorides. They also have a low amount of free
sulfur dioxide but a high total sulfur dioxide. These wines are relatively dense with a high pH
level, and they have low sulphates and alcohol content.

The same interpretation can be applied to the other two centroids. Each represents a different
group of wines with their own unique characteristics.

<img width="797" height="182" alt="image" src="https://github.com/user-attachments/assets/4fd40bf8-0019-4ee0-afd0-e354773ea437" />

Figure 5. Centroids

## Conclusion

### Summary

To conclude, the K-means clustering results for the Vinho Verde wine dataset reveal some
natural groupings, but they are not well-separated. While the clusters exhibit moderate
cohesion, the overall variability remains relatively high. To improve the prediction of wine
quality, further exploration and refinement of the clustering approach may be necessary.

### Limitations & Improvement areas 
Some limitations of the K-means clustering algorithm include:

**Choosing k Manually**: Selecting the optimal number of clusters can be challenging.

**Handling Varying Cluster Sizes and Densities**: K-means assumes that clusters are spherical
and have similar sizes. In real-world data, clusters can have irregular shapes and varying
densities. K-means may struggle to handle such cases, resulting in less accurate clusters.

**Clustering Outliers**: Outliers can significantly impact k-means. Centroids may be dragged by
outliers, leading to suboptimal cluster assignments.

## Appendix

**Data Transforming**:

```
#Drop the 'type' column from the DataFrame 'cust_df'

df = cust_df.drop('type', axis=1)
```

**Data Normalization**:

```
#Import StandardScaler from sklearn.preprocessing

from sklearn.preprocessing import StandardScaler
```

```
#Extract features (excluding the first column) from the DataFrame 'df'

X = df.values[:, 1:]
```

```
#Replace any NaN values with zeros

X = np.nan_to_num(X)
```

```
#Standardize the features using StandardScaler

X = StandardScaler().fit_transform(X)


#Display the standardized features

X
```

**K-Means Clustering**:

```
#Fit K-means model

clusterNum = 3

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

Labels:

labels = k_means.labels_

df["Clus_km"] = labels

Centroids:

print("Centroids:\n", k_means.cluster_centers_)
```

**Silhouette Coefficient**:

```
#Import the silhouette_score function from sklearn.metrics

from sklearn.metrics import silhouette_score
```

```
#Calculate the silhouette coefficient

silhouette_coef = silhouette_score(X, labels)
```

```
#Print the result

print(f"Silhouette Coefficient: {silhouette_coef:.2f}")
```

```

Intra-cluster Distance:

#Calculate intra-cluster distances

intra_cluster_distances = []

for cluster_id in range(clusterNum):

cluster_points = X[labels == cluster_id]

centroid = k_means.cluster_centers_[cluster_id]

distances = np.linalg.norm(cluster_points - centroid, axis=1)

intra_cluster_distances.append(distances.mean())
```

```
#Print intra-cluster distances

for cluster_id, distance in enumerate(intra_cluster_distances):

print(f"Cluster {cluster_id}: Intra-cluster distance = {distance:.2f}")
```

```
Inter-cluster Distance:

#Calculate the inter-cluster distances (inertia) using k-means

inter_cluster_distances = k_means.inertia_
```

```
Print the result

print(f"Inter-cluster Distance (Inertia): {inter_cluster_distances:.2f}")
```

## References

Bhardwaj, A. (2020, May 27). Silhouette coefficient : Validating clustering techniques.

Medium. https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c

Education Ecosystem (LEDU). (2018, September 12). Understanding K-means clustering in

machine learning. Medium. https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa

K-means advantages and disadvantages. (n.d.). Google for Developers.

https://developers.google.com/machine-learning/clustering/algorithm/advantages-disadvantages

ML | Intercluster and Intracluster distance. (2021, August 8). GeeksforGeeks.

https://www.geeksforgeeks.org/ml-intercluster-and-intracluster-distance/

Ryzhkov, E. (2020, July 23). 5 stages of data Preprocessing for K-means clustering. Medium.

https://medium.com/@evgen.ryzhkov/5-stages-of-data-preprocessing-for-k-means-clustering-b755426f

Wine quality data set (Red & white wine). (n.d.). Kaggle: Your Machine Learning and Data

Science Community. https://www.kaggle.com/datasets/ruthgn/wine-quality-data-set-red-white-wine?
