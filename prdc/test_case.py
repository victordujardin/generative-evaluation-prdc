
import numpy as np
from sklearn.preprocessing import normalize
from prdc import compute_prdc

# Example feature vectors for real and fake images (10,000 samples, 2048 features each)
# Generate real and fake features from a normal distribution with mean 1 and std 1
real_features = np.random.normal(loc=1, scale=1, size=(100, 2048)).astype(np.float32)
fake_features = np.random.normal(loc=2, scale=1.5, size=(100, 2048)).astype(np.float32)


weigths = np.random.randint(1, 101, size=100)


population_size = weigths.sum()

weigths = weigths/population_size



# Normalize the features
real_features = normalize(real_features)
fake_features = normalize(fake_features)

# Define nearest_k, usually set to a small number like 5
nearest_k = 5

# Compute PRDC metrics
metrics = compute_prdc(real_features, fake_features, nearest_k, population_size=100, sample_size = 100, weights=None)

# Display results
print("PRDC Metrics:")
print("Precision: {:.4f}".format(metrics['precision']))
print("Recall: {:.4f}".format(metrics['recall']))
print("Density: {:.4f}".format(metrics['density']))
print("new Density: {:.4f}".format(metrics['new_density']))
print("Coverage: {:.4f}".format(metrics['coverage']))




