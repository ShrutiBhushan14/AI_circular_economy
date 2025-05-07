
# Load synthetic dump data with lat, lon, and dump_weight
df = pd.read_csv('dump_data.csv')  # Assumed CSV with 'latitude', 'longitude', 'weight'

X = df[['latitude', 'longitude']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 10), inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal Clusters')
plt.show()

# Clustering with optimal number of zones (assume 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['zone'] = kmeans.fit_predict(X_scaled)

# Visualize clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x='longitude', y='latitude', hue='zone', data=df, palette='Set1')
plt.title('Waste Dump Zones')
plt.show()
