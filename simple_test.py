import numpy as np
from sklearn.preprocessing import normalize
from prdc import compute_prdc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit  # Fonction sigmoïde



# Real samples coordinates (approximating based on the provided image)
real_samples = np.array([
    [3.1, 2.0], 
    [3.6, 1.8], 
    [3.5, 2.4], 
    [3.2, 2.1], 
    [3.4, 2.4], 
    [3.8, 2.5], 
    [5.0, 2.5]
], dtype=np.float32)

# Fake samples coordinates (approximating based on the provided image)
fake_samples = np.array([
    [4.0, 2.5], 
    [4.5, 2.4], 
    [5.2, 2.4], 
    [4.6, 2.7], 
    [5.0, 2.6]
    ], dtype=np.float32)


beta_0 = 0.0  # Intercept
beta_1 = -1.0   # Coefficient de la première variable
beta_2 = 2.0   # Coefficient de la deuxième variable
beta = np.array([beta_0, beta_1, beta_2])

# 3. Calculer la probabilité à l'aide de la fonction logistique
X_with_intercept = np.column_stack([np.ones(real_samples.shape[0]), real_samples])  # Ajout d'une colonne de 1 pour l'interception
Y_with_intercept = np.column_stack([np.ones(fake_samples.shape[0]), fake_samples])  # Ajout d'une colonne de 1 pour l'interception
p = expit(np.dot(X_with_intercept, beta))  # Fonction sigmoïde
p_star = expit(np.dot(Y_with_intercept, beta))  # Fonction sigmoïde
w = 1/p 
w_star = 1/p_star 



# Creating a DataFrame for the real and fake samples
real_samples_df = pd.DataFrame(real_samples, columns=["x", "y"])
fake_samples_df = pd.DataFrame(fake_samples, columns=["x", "y"])

# Adding a label to distinguish real from fake samples
real_samples_df['label'] = 'real'
fake_samples_df['label'] = 'fake'

# Concatenating both DataFrames
samples_df = pd.concat([real_samples_df, fake_samples_df], ignore_index=True)



plt.figure(figsize=(6, 6))

# Plot real samples and annotate with weights
for i, (x, y) in enumerate(real_samples):
    plt.scatter(x, y, color='purple', label='Real Samples' if i == 0 else "", s=100, alpha=0.6)
    plt.annotate(f"{w[i]:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

# Plot fake samples and annotate with weights
for i, (x, y) in enumerate(fake_samples):
    plt.scatter(x, y, color='green', label='Fake Samples' if i == 0 else "", s=100)
    plt.annotate(f"{w_star[i]:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

# Label the axes
plt.xlabel("X")
plt.ylabel("Y")

# Adding a title and legend
plt.title("Plot of Real and Fake Samples")
plt.legend()

# Show the plot
plt.grid(True)
plt.savefig("figure1")



# Define nearest_k, usually set to a small number like 5
nearest_k = 2

# Extract only numerical data for compute_prdc function
real_features = real_samples_df[['x', 'y']].to_numpy()
fake_features = fake_samples_df[['x', 'y']].to_numpy()

# Compute PRDC metrics
metrics = compute_prdc(real_features, fake_features, nearest_k, weights=w , weights_star=w_star)

# Display results
print("PRDC Metrics:")
print("Precision: {:.4f}".format(metrics['precision']))
print("Density_Naeem: {:.4f}".format(metrics['density_Naeem']))
print("Density_Hugues: {:.4f}".format(metrics['density_Hugues']))
#Attention pour la weighted density, les poids ne doivent pas sommer à 1
print("Weighted Density: {:.4f}".format(metrics['weighted_density']))
print("Weighted Coverage: {:.4f}".format(metrics['weighted_coverage']))
