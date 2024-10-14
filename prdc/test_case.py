
import numpy as np
from sklearn.preprocessing import normalize
from prdc import compute_prdc
import pandas as pd


real_features = pd.read_csv("pred_original.csv")
fake_features = pd.read_csv("pred_pred.csv")


# Keep only numeric columns
real_features = real_features.select_dtypes(include=['number'])
fake_features = fake_features.select_dtypes(include=['number'])

# Reset indices
real_features = real_features.reset_index(drop=True)
fake_features = fake_features.reset_index(drop=True)

# Drop rows with NaN values
real_features = real_features.dropna()
fake_features = fake_features.dropna()

# Align DataFrames by their indices
min_length = min(len(real_features), len(fake_features))
real_features = real_features.iloc[:min_length]
fake_features = fake_features.iloc[:min_length]

# Update weights array
weights = np.ones(min_length, dtype=np.float32)


# Define nearest_k, usually set to a small number like 5
nearest_k = 5

# Compute PRDC metrics
metrics = compute_prdc(real_features, fake_features, nearest_k, population_size=100, sample_size = 100, weights=weights)

# Display results
print("PRDC Metrics:")
# print("Precision: {:.4f}".format(metrics['precision']))
# print("Recall: {:.4f}".format(metrics['recall']))
print("Density_Naeem: {:.4f}".format(metrics['density_Naeem']))
print("Density_Hugues: {:.4f}".format(metrics['density_Hugues']))
print("Weighted Density: {:.4f}".format(metrics['weighted_density']))
# print("Coverage: {:.4f}".format(metrics['coverage']))




