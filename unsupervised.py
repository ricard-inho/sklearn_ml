import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('Livreur_Rejections.csv')

# Handle missing values if necessary
data = data.dropna()

# Feature selection (including only numerical features for clustering)
features = ['Avg_Temp(C)', 'Precipitation(ml)', 'Wind_Speed(km/hr)', 'lat', 'lng']
X = data[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_summary = data.groupby('Cluster').agg({
    'Accepted': 'mean',
    'Rejected': 'mean',
    'Avg_Temp(C)': 'mean',
    'Precipitation(ml)': 'mean',
    'Wind_Speed(km/hr)': 'mean',
    'lat': 'mean',
    'lng': 'mean'
}).reset_index()

print(cluster_summary)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='lat', y='lng', hue='Cluster', palette='viridis')
plt.title('Clusters by Location')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.savefig("cluster.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Cluster', y='Rejected')
plt.title('Rejections by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Rejections')
plt.savefig("cluster_reject.png")
plt.show()
