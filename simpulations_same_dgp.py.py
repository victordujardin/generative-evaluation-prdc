import numpy as np
from scipy.special import expit  # Fonction sigmoïde
import pandas as pd

# 1. Générer des variables explicatives (par exemple, 10000 échantillons, 5 variables explicatives)
np.random.seed(42)
X = np.random.normal(0, 1, (10000, 5))

# 2. Définir les coefficients de la régression logistique (1 intercept + 5 coefficients)
beta_0 = -0.5  # Intercept
beta_1 = 1.0   # Coefficient de la première variable
beta_2 = 2.0   # Coefficient de la deuxième variable
beta_3 = 0.5   # Coefficient de la troisième variable
beta_4 = -1.5  # Coefficient de la quatrième variable
beta_5 = 0.75  # Coefficient de la cinquième variable

# Créer un tableau de coefficients (intercept + coefficients des 5 variables)
beta = np.array([beta_0, beta_1, beta_2, beta_3, beta_4, beta_5])

# 3. Calculer la probabilité à l'aide de la fonction logistique
X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])  # Ajout d'une colonne de 1 pour l'interception
p = expit(np.dot(X_with_intercept, beta))  # Fonction sigmoïde
w = 1/p

# 4. Générer la matrice aléatoire (5x32)
matrice_aleatoire = np.ones((5, 32))

# 5. Définir la matrice de covariance (par exemple, utiliser la matrice identité)
covariance_matrix = np.identity(32)  # Covariance matrix de dimension 32x32

# 6. Initialiser une liste pour stocker les données générées pour chaque observation
data_multivariees = []
fake_multivariees = []



# 7. Boucler sur chaque observation de X pour générer des données multivariées
for i in range(X.shape[0]):
    # Calculer la moyenne pour l'observation i
    mean_observation = np.dot(matrice_aleatoire.T, X[i])
    
    # Générer des données multivariées pour l'observation i
    generated_data = np.random.multivariate_normal(mean_observation, covariance_matrix)
    fake_data = np.random.multivariate_normal(mean_observation, covariance_matrix)
    
    # Ajouter les données générées à la liste
    data_multivariees.append(generated_data)
    fake_multivariees.append(fake_data)

# Convertir la liste en DataFrame
data_multivariees_df = pd.DataFrame(data_multivariees)

# Ajouter les colonnes 'w' et 'p' au DataFrame original
data = pd.DataFrame(data_multivariees_df)
data['w'] = w
data['p'] = p



# Sample 100 rows based on the probability 'p'
sampled_data = data.sample(n=100, weights='p', random_state=42)





# # Initialize empty lists for each metric
# precision_list = []
# recall_list = []
# density_Hugues_list = []
# coverage_list = []
# density_Naeem_list = []
# weighted_density_list = []
# weighted_coverage_list = []

# for k in range(1, 100):
#     # Assume compute_prdc is a function that returns a dictionary with the required metrics
#     metrics = compute_prdc(sampled_data.iloc[:, :5], fake_data, k, 10000, 100, w)
    
#     # Extract metrics from the dictionary
#     precision = metrics['precision']
#     recall = metrics['recall']
#     density_Hugues = metrics['density_Hugues']
#     coverage = metrics['coverage']
#     density_Naeem = metrics['density_Naeem']
#     weighted_density = metrics['weighted_density']
#     weighted_coverage = metrics['weighted_coverage']
    
#     # Append each metric to its corresponding list
#     precision_list.append(precision)
#     recall_list.append(recall)
#     density_Hugues_list.append(density_Hugues)
#     coverage_list.append(coverage)
#     density_Naeem_list.append(density_Naeem)
#     weighted_density_list.append(weighted_density)
#     weighted_coverage_list.append(weighted_coverage)
    
#     # Print the metrics for each iteration
#     print(f"Nearest neighbor {k}:")
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(f"Density Hugues: {density_Hugues}")
#     print(f"Density Naeem: {density_Naeem}")
#     print(f"Weighted Density: {weighted_density}")
#     print(f"Coverage: {coverage}")
#     print(f"Weighted Coverage: {weighted_coverage}")
#     print("-" * 30)
