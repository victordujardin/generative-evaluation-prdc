
import numpy as np
from sklearn.preprocessing import normalize
from prdc import compute_prdc

# Example feature vectors for real and fake images (10,000 samples, 2048 features each)
real_features = np.random.randn(100, 2048).astype(np.float32)
fake_features = np.random.randn(100, 2048).astype(np.float32)

# Normalize the features
real_features = normalize(real_features)
fake_features = normalize(fake_features)

# Define nearest_k, usually set to a small number like 5
nearest_k = 5

# Compute PRDC metrics
metrics = compute_prdc(real_features, fake_features, nearest_k)

# Display results
print("PRDC Metrics:")
print("Precision: {:.4f}".format(metrics['precision']))
print("Recall: {:.4f}".format(metrics['recall']))
print("Density: {:.4f}".format(metrics['density']))
print("Coverage: {:.4f}".format(metrics['coverage']))


