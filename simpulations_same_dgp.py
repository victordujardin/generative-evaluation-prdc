import numpy as np
from scipy.special import expit  # Fonction sigmoïde
import pandas as pd
import matplotlib.pyplot as plt
from prdc import compute_prdc
from scipy.stats import wishart

# 1. Générer des variables explicatives 
np.random.seed(42)

n = 800
m = 2000

lowrank = 5

X = np.random.normal(0, 1, (m, lowrank))
Y = np.random.normal(0, 1, (m, lowrank))

# 2. Définir les coefficients de la régression logistique (1 intercept + 5 coefficients)
beta_0 = 0.0  # Intercept
beta_1 = 5.0   # Coefficient de la première variable
beta_2 = -5.0   # Coefficient de la deuxième variable
beta_3 = -5.0   # Coefficient de la troisième variable
beta_4 = -5.0  # Coefficient de la quatrième variable
beta_5 = 5.0  # Coefficient de la cinquième variable

# Créer un tableau de coefficients (intercept + coefficients des 5 variables)
beta = np.array([beta_0, beta_1, beta_2, beta_3, beta_4, beta_5])

# 3. Calculer la probabilité à l'aide de la fonction logistique
X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])  # Ajout d'une colonne de 1 pour l'interception
Y_with_intercept = np.column_stack([np.ones(Y.shape[0]), Y])  # Ajout d'une colonne de 1 pour l'interception
p = expit(np.dot(X_with_intercept, beta))  # Fonction sigmoïde
p_star = expit(np.dot(Y_with_intercept, beta))  # Fonction sigmoïde
w = 1/p 
w_star = 1/p_star 

datatest = pd.DataFrame(X)
datatest['p'] = p
sampletest = datatest.sample(n=n, weights='p', random_state=42)


plt.figure(figsize=(6, 4))
plt.scatter(sampletest.iloc[:, 0], sampletest.iloc[:, 1], color='blue', s=10)  # 's' sets the size of the dots
# plt.show()

# 4. Générer la matrice aléatoire (5x32)
matrice_aleatoire =  np.random.randn(lowrank, 32)

# 5. Définir la matrice de covariance 
covariance_matrix = np.dot(np.dot(matrice_aleatoire.T, np.cov(X, rowvar=False)), matrice_aleatoire) 
covariance_matrix_fake = np.dot(np.dot(matrice_aleatoire.T, np.cov(Y, rowvar=False)), matrice_aleatoire) 
# 6. Initialiser une liste pour stocker les données générées pour chaque observation
data_multi = []
fake_multi = []

m = Y.shape[0]

# 7. Boucler sur chaque observation de X pour générer des données multivariées
for i in range(X.shape[0]):
    # Calculer la moyenne pour l'observation i
    mean_observation = np.dot(matrice_aleatoire.T, X[i])
    para_poisson = abs(X[i].sum())

    mean_observation_fake = np.dot(matrice_aleatoire.T, Y[i])
    para_poisson_fake = abs(Y[i].sum())


    multinomial_probs = expit(X[i])  # Utiliser la fonction sigmoïde pour obtenir des probabilités
    multinomial_probs /= multinomial_probs.sum()  # Normaliser pour que la somme soit 1

    multinomial_probs_fake = expit(Y[i])  # Utiliser la fonction sigmoïde pour obtenir des probabilités
    multinomial_probs_fake /= multinomial_probs_fake.sum()  # Normaliser pour que la somme soit 1
    
    
    # Générer des données multivariées pour l'observation i
    generated_data = np.random.multivariate_normal(mean_observation, covariance_matrix)
    fake_data = np.random.multivariate_normal(mean_observation_fake, covariance_matrix_fake)

    lambda_exp = np.exp(abs(np.dot(X[i], np.array([0.2, -0.1, 0.4, -0.3, 0.1])))) #, 0.4, -0.3, 0.1
    lambda_exp_fake = np.exp(abs(np.dot(Y[i], np.array([0.2, -0.1, 0.4, -0.3, 0.1]))))#, 0.4, -0.3, 0.1



    # Ajouter les données générées à la liste
    data_multi.append(list(generated_data)+ list(np.random.poisson(para_poisson, 100)) + list(np.random.multinomial(1, multinomial_probs)) + [np.random.exponential(lambda_exp)])
    fake_multi.append(list(fake_data)+ list(np.random.poisson(para_poisson_fake, 100)) + list(np.random.multinomial(1, multinomial_probs_fake))+ [np.random.exponential(lambda_exp_fake)])



# Convertir la liste en DataFrame
data_multi_df = pd.DataFrame(data_multi)
fake_multi_df = pd.DataFrame(fake_multi)



# Ajouter les colonnes 'w' et 'p' au DataFrame original
data = pd.DataFrame(data_multi_df)
data['w'] = w
data['p'] = p




# Sample 100 rows based on the probability 'p'
sampled_data = data.sample(n=n, weights='p', random_state=42)




# Initialize empty lists for each metric
precision_list = []
recall_list = []
density_Hugues_list = []
coverage_list = []
density_Naeem_list = []
weighted_density_list = []
weighted_coverage_list = []


K_lim = n - 1

for k in range(1, K_lim):
    # Assume compute_prdc is a function that returns a dictionary with the required metrics
    metrics = compute_prdc(sampled_data.iloc[:, :-2], fake_multi_df, k, sampled_data["w"], weights_star=None, normalized = False)
    
    # Extract metrics from the dictionary
    precision = metrics['precision']
    recall = metrics['recall']
    density_Hugues = metrics['density_Hugues']
    coverage = metrics['coverage']
    density_Naeem = metrics['density_Naeem']
    weighted_density = metrics['weighted_density']
    weighted_coverage = metrics['weighted_coverage']
    
    # Append each metric to its corresponding list
    precision_list.append(precision)
    recall_list.append(recall)
    density_Hugues_list.append(density_Hugues)
    coverage_list.append(coverage)
    density_Naeem_list.append(density_Naeem)
    weighted_density_list.append(weighted_density)
    weighted_coverage_list.append(weighted_coverage)
    
    # Print the metrics for each iteration
    print()
    print(f"Nearest neighbor {k}:")
    print()
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"Density Hugues: {density_Hugues}")
    print(f"Density Naeem: {density_Naeem}")
    print(f"Weighted Density: {weighted_density}")
    print(f"Coverage: {coverage}")
    print(f"Weighted Coverage: {weighted_coverage}")
    # print("-" * 30)


k_values = range(1, K_lim)

plt.figure(figsize=(10, 6))
plt.plot(k_values, density_Naeem_list, label='Density Naeem', marker='o')
plt.plot(k_values, weighted_density_list, label='Weighted Density', marker='x')
plt.xlabel('k')
plt.ylabel('Density')
plt.title(f'Density Naeem and Weighted Density with respect to k, for n = {n} and m = {m}')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("density_convergence_plot_same_dgp.png") 


plt.figure(figsize=(10, 6))
plt.plot(k_values, coverage_list, label='Coverage Naeem', marker='o')
plt.plot(k_values, weighted_coverage_list, label='Weighted Coverage', marker='x')
plt.xlabel('k')
plt.ylabel('Density')
plt.title(f'Coverage Naeem and Weighted Coverage with respect to k, for n = {n} and m = {m}')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("coverage_convergence_plot_same_dgp.png") 