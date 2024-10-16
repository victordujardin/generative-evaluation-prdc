
import numpy as np
from sklearn.preprocessing import normalize
from prdc import compute_prdc
import pandas as pd


real_features = pd.read_csv("pred_original.csv")
fake_features = pd.read_csv("pred_pred.csv")


# Reset indices
real_features = real_features.reset_index(drop=True)
fake_features = fake_features.reset_index(drop=True)

# Drop rows with NaN values
real_features = real_features.dropna()
fake_features = fake_features.dropna()



# List of non-numerical columns
non_numerical_columns = [
    'dco1yb_p_lc', 'dcobi_p_lc', 'dcosu_p_lc', 'dcowo_p_lc', 'dedle_p_lc',
    'dedlefat_p_lc', 'dedlemot_p_lc', 'dna_p_lc', 'dqrffix_p_lc', 'dysrs_p_lc',
    'locna1yb_p_lc', 'locna_p_lc', 'locnalj_p_lc', 'locnasj_p_lc'
]

# Encode the non-numerical variables with dummy variables
real_features_encoded = pd.get_dummies(real_features[non_numerical_columns])
fake_features_encoded = pd.get_dummies(fake_features[non_numerical_columns])

# Ensure both DataFrames have the same dummy columns
real_features_encoded, fake_features_encoded = real_features_encoded.align(fake_features_encoded, join='outer', axis=1, fill_value=0)


# Filter the DataFrames to include only numeric columns
real_features = real_features.select_dtypes(include=['number'])
fake_features = fake_features.select_dtypes(include=['number'])


# Integrate the encoded dummy variables into the numerical DataFrames
real_features = pd.concat([real_features, real_features_encoded], axis=1)
fake_features = pd.concat([fake_features, fake_features_encoded], axis=1)

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
print("Coverage: {:.4f}".format(metrics['coverage']))
print("Weighted Coverage: {:.4f}".format(metrics['weighted_coverage']))




