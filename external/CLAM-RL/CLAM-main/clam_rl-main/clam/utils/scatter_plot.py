import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import numpy as np
# Function to parse the embeddings into lists of floats
def parse_embeddings(row):
    embeddings = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", row)]
    return embeddings if len(embeddings) == 20 else [None] * 20

# Load your dataset
file_path = 'all_data_policy_10.csv'  # Replace with your dataset file path
data = pd.read_csv(file_path)

# Apply the parsing function to the 'embeddings' column
data['embeddings'] = data['embeddings'].apply(parse_embeddings)

# Filter out rows where embeddings are not 20-dimensional
filtered_data = data[data['embeddings'].apply(lambda x: None not in x)]

# Remove the last row (special points)
filtered_data = filtered_data[:-1]
# Extract embeddings as a list of lists
filtered_embeddings = filtered_data['embeddings'].tolist()
print("filtered_embeddings",np.array(filtered_embeddings).min(),np.array(filtered_embeddings).max())
# Special points - Extracted from the last row of the dataset
last_row_values = re.findall(r"[-+]?\d*\.\d+|\d+", data.iloc[-1, 0])
special_points = [float(value) for value in last_row_values]
print("special_points",np.array(special_points).min(),np.array(special_points).max())

# Select the dimensionality reduction method
method = 'tsne'  # Change to 'tsne' for t-SNE

# Apply UMAP or t-SNE
if method == 'umap':
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
else:  # t-SNE
    reducer = TSNE(n_components=2, random_state=42)

embeddings_2d = reducer.fit_transform(filtered_embeddings)

# Creating a scatter plot
plt.figure(figsize=(12, 10))
sns.scatterplot(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    hue=filtered_data['policy_id'].tolist(),
    palette=sns.color_palette("hsv", len(filtered_data['policy_id'].unique())),
    legend='full'
)

# Overlaying the special points
for idx, value in enumerate(special_points):
    plt.scatter(
        [value],
        [0],  # Arbitrarily setting y-coordinate to 0 for visualization
        color='black',
        s=1000,  # size of the marker
        marker='*',  # star shaped marker
        label=f'Special Point {idx+1}' if idx == 0 else None  # label only once
    )

plt.xlabel(f'{method.upper()} Dimension 1')
plt.ylabel(f'{method.upper()} Dimension 2')
plt.title(f'2D Scatter Plot of Embeddings Using {method.upper()} with Special Points Highlighted')
plt.legend(title='Policy ID')
plt.show()
