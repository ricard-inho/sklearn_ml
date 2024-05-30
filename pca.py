import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Livreur_Rejections.csv')

# Handle missing values if necessary
data = data.dropna()

# Feature selection (including only numerical features for PCA)
features = ['Avg_Temp(C)', 'Precipitation(ml)', 'Wind_Speed(km/hr)', 'lat', 'lng']
X = data[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_

# Plot the explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.savefig("pca_var.png")
plt.show()

# Cumulative explained variance
cumulative_explained_variance = explained_variance.cumsum()

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.axhline(y=0.90, color='g', linestyle='-')
plt.savefig("pca_cumm_var.png")
plt.show()

# Print the explained variance for each component
for i, var in enumerate(explained_variance):
    print(f'Principal Component {i+1}: {var:.2f}')

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Add back the non-feature columns for context
pc_df = pd.concat([pc_df, data[['Accepted', 'Rejected', 'Delivery_Date']]], axis=1)

print(pc_df.head())
