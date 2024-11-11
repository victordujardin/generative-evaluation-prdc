import numpy as np
from scipy.special import expit  # Fonction sigmoïde
import pandas as pd
import matplotlib.pyplot as plt
from prdc import compute_prdc
from scipy.stats import wishart
from sampling import sequential_weighted_sample

# 1. Générer des variables explicatives 
seed = 42
np.random.seed(seed)


m = 100
n = 10
num_runs  = 1000
K_lim = n - 1
k_values = list(range(1, K_lim))
density_Naeem_all = {k: [] for k in range(1, K_lim)}
weighted_density_all = {k: [] for k in range(1, K_lim)}
weighted_density_threshold_all = {k: [] for k in range(1, K_lim)}
coverage_all = {k: [] for k in range(1, K_lim)}
weighted_coverage_all = {k: [] for k in range(1, K_lim)}

for i in range(num_runs):


    lowrank = 10



    X = np.random.multivariate_normal(np.zeros(lowrank), np.identity(lowrank), size = m)
    Y = np.random.multivariate_normal(np.zeros(lowrank), np.identity(lowrank), size = m)
    X = pd.DataFrame(X, columns=[f'X_{j}' for j in range(lowrank)])
    Y = pd.DataFrame(Y, columns=[f'Y_{j}' for j in range(lowrank)])

    # 2. Définir les coefficients de la régression logistique (1 intercept + 5 coefficients)
    beta_0 = 0.0  # Intercept
    beta_1 = 5.0   # Coefficient de la première variable
    beta_2 = -5.0   # Coefficient de la deuxième variable
    beta_3 = -5.0   # Coefficient de la troisième variable
    beta_4 = -5.0  # Coefficient de la quatrième variable
    beta_5 = 5.0  # Coefficient de la cinquième variable
    beta_6 = 5.0   # Coefficient de la première variable
    beta_7 = -5.0   # Coefficient de la deuxième variable
    beta_8 = -5.0   # Coefficient de la troisième variable
    beta_9 = -5.0  # Coefficient de la quatrième variable
    beta_10 = 5.0  # Coefficient de la cinquième variable

    # Créer un tableau de coefficients (intercept + coefficients des 5 variables)
    beta = np.array([beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8, beta_9, beta_10])

    # 3. Calculer la probabilité à l'aide de la fonction logistique
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])  # Ajout d'une colonne de 1 pour l'interception
    Y_with_intercept = np.column_stack([np.ones(Y.shape[0]), Y])  # Ajout d'une colonne de 1 pour l'interception
    p = expit(np.dot(X_with_intercept, beta))  # Fonction sigmoïde
    p_star = expit(np.dot(Y_with_intercept, beta))  # Fonction sigmoïde
    w = 1/p 
    w_star = 1/p_star 



    data_multi = X.copy()
    fake_multi = Y.copy()

    # Calculer les moyennes par ligne pour X et Y
    data_multi['mean_X'] = data_multi.mean(axis=1)
    fake_multi['mean_Y'] = fake_multi.mean(axis=1)


    #Générer la matrice aléatoire (5x32)
    matrice_aleatoire =  np.random.randn(lowrank, 32)

    # Définir la matrice de covariance 
    covariance_matrix = np.dot(np.dot(matrice_aleatoire.T, np.cov(X, rowvar=False)), matrice_aleatoire) 
    covariance_matrix_fake = np.dot(np.dot(matrice_aleatoire.T, np.cov(Y, rowvar=False)), matrice_aleatoire) 

    # Poisson distribution
    n_pois = 10
    data_multi_poisson = np.random.poisson(lam=abs(data_multi['mean_X']).values[:, np.newaxis], size=(m, n_pois))
    poisson_columns = [f'poisson_{j}' for j in range(n_pois)]
    data_multi[poisson_columns] = data_multi_poisson

    fake_multi_poisson = np.random.poisson(lam=abs(fake_multi['mean_Y']).values[:, np.newaxis], size=(m, n_pois))
    poisson_fake_columns = [f'poisson_fake_{j}' for j in range(n_pois)]
    fake_multi[poisson_fake_columns] = fake_multi_poisson


    # 2. Multinomial Distribution


    X_np = X.values


    multinomial_probs = expit(X_np)  # Apply sigmoid to get probabilities for each column
    multinomial_probs /= multinomial_probs.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1
    multinomial_samples = np.array([np.random.multinomial(1, p) for p in multinomial_probs])
    multinomial_columns = [f'multinomial_{j}' for j in range(X_np.shape[1])]
    data_multi[multinomial_columns] = multinomial_samples

    # Similar modification for Y
    Y_np = Y.values
    multinomial_probs_fake = expit(Y_np)  # For Y
    multinomial_probs_fake /= multinomial_probs_fake.sum(axis=1, keepdims=True)
    multinomial_samples_fake = np.array([np.random.multinomial(1, p) for p in multinomial_probs_fake])
    multinomial_fake_columns = [f'multinomial_fake_{j}' for j in range(Y_np.shape[1])]
    fake_multi[multinomial_fake_columns] = multinomial_samples_fake

    # 3. Multivariate Normal Distribution
    # Generate mean vectors and covariance matrices for all rows in one go
    mean_observations = np.dot(X, matrice_aleatoire)  # Shape (m, 32)
    mean_observations_fake = np.dot(Y, matrice_aleatoire)  # Shape (m, 32)

    # Generate multivariate normal samples for all observations at once
    multivariate_normal_samples = np.random.multivariate_normal(
        mean=np.zeros(mean_observations.shape[1]), cov=covariance_matrix, size=m
    ) + mean_observations
    multivariate_columns = [f'multivariate_{j}' for j in range(mean_observations.shape[1])]
    data_multi[multivariate_columns] = multivariate_normal_samples

    multivariate_normal_samples_fake = np.random.multivariate_normal(
        mean=np.zeros(mean_observations_fake.shape[1]), cov=covariance_matrix_fake, size=m
    ) + mean_observations_fake
    multivariate_fake_columns = [f'multivariate_fake_{j}' for j in range(mean_observations_fake.shape[1])]
    fake_multi[multivariate_fake_columns] = multivariate_normal_samples_fake

    # 4. Exponential Distribution
    lambda_exp = np.exp(np.abs(np.dot(X, np.array([0.2, -0.1, 0.4, -0.3, 0.1, 0.2, -0.1, 0.4, -0.3, 0.1]))))
    exponential_samples = np.random.exponential(scale=lambda_exp)  # Scale = 1/lambda
    data_multi['exponential'] = exponential_samples

    lambda_exp_fake = np.exp(np.abs(np.dot(Y, np.array([0.2, -0.1, 0.4, -0.3, 0.1, 0.2, -0.1, 0.4, -0.3, 0.1]))))
    exponential_samples_fake = np.random.exponential(scale=lambda_exp_fake)
    fake_multi['exponential_fake'] = exponential_samples_fake













    # Ajouter les colonnes 'w' et 'p' au DataFrame original
    data = pd.DataFrame(data_multi)
    data['w'] = w
    data['p'] = p

    fakedata = pd.DataFrame(fake_multi)
    fakedata['w_star'] = w_star
    fakedata['p_star'] = p_star




    # Sample n rows based on the probability 'p'

    # sampled_data = sequential_weighted_sample(data, weights=data["p"], n = n , seed = 42)
    sampled_data = data.sample(n=n, weights='p', random_state=seed) # bien sans remplacement, #écrire en tex de manière théorique.
    sampled_data_fake = fakedata.sample(n=n, weights='p_star', random_state=seed) 



    # Initialize empty lists for each metric
    precision_list = []
    recall_list = []
    density_Hugues_list = []
    coverage_list = []
    density_Naeem_list = []
    weighted_density_list = []
    weighted_coverage_list = []


    for k in range(1, K_lim):
        # Assume compute_prdc is a function that returns a dictionary with the required metrics
        metrics = compute_prdc(sampled_data.iloc[:, :-2], sampled_data_fake.iloc[: , :-2], k, weights=sampled_data["w"], weights_star=sampled_data_fake["w_star"], normalized = False, weight_threshold=k*sampled_data["w"].mean())
        
        # Extract required metrics
        density_Naeem = metrics.get('density_Naeem', np.nan)
        weighted_density = metrics.get('weighted_density', np.nan)
        coverage = metrics.get('coverage', np.nan)
        weighted_coverage = metrics.get('weighted_coverage', np.nan)
        weighted_density_threshold = metrics.get('weighted_density_threshold', np.nan)
        # Extract other metrics if needed

        # Store the metrics
        density_Naeem_all[k].append(density_Naeem)
        weighted_density_all[k].append(weighted_density)
        coverage_all[k].append(coverage)
        weighted_coverage_all[k].append(weighted_coverage)
        weighted_density_threshold_all[k].append(weighted_density_threshold)

    print(f"Run {i + 1} completed.\n")




# After all runs, compute mean and variance for each metric at each k
mean_density_Naeem = {k: np.mean(v) for k, v in density_Naeem_all.items()}
var_density_Naeem = {k: np.var(v) for k, v in density_Naeem_all.items()}

mean_weighted_density = {k: np.mean(v) for k, v in weighted_density_all.items()}
var_weighted_density = {k: np.var(v) for k, v in weighted_density_all.items()}

mean_coverage = {k: np.mean(v) for k, v in coverage_all.items()}
var_coverage = {k: np.var(v) for k, v in coverage_all.items()}

mean_weighted_coverage = {k: np.mean(v) for k, v in weighted_coverage_all.items()}
var_weighted_coverage = {k: np.var(v) for k, v in weighted_coverage_all.items()}


mean_weighted_density_threshold = {k: np.mean(v) for k, v in weighted_density_threshold_all.items()}
var_weighted_density_threshold = {k: np.var(v) for k, v in weighted_density_threshold_all.items()}




# Convert means and variances to lists in the order of k_values
density_naeem_means = [mean_density_Naeem[k] for k in k_values]
density_naeem_vars = [var_density_Naeem[k] for k in k_values]

weighted_density_means = [mean_weighted_density[k] for k in k_values]
weighted_density_vars = [var_weighted_density[k] for k in k_values]

weighted_density_threshold_means = [mean_weighted_density_threshold[k] for k in k_values]
weighted_density_threshold_vars = [var_weighted_density_threshold[k] for k in k_values]


mse_density_naeem = [var +(mean - 1)**2 for var, mean in zip(density_naeem_vars, density_naeem_means)]
mse_weighted_density = [var + (mean-1)**2 for var, mean in zip(weighted_density_vars, weighted_density_means)]
mse_weighted_density_threshold = [var + (mean-1)**2 for var, mean in zip(weighted_density_threshold_vars, weighted_density_threshold_means)]






# Calculate standard deviations for variance regions
density_naeem_stds = np.sqrt(density_naeem_vars)
weighted_density_stds = np.sqrt(weighted_density_vars)
weighted_density_threshold_stds = np.sqrt(weighted_density_threshold_vars)

# Now, call the plotting functions
plot_graphs.plot_density_metrics(
    k_values,
    density_naeem_means,
    density_naeem_stds,
    weighted_density_means,
    weighted_density_stds,
    weighted_density_threshold_means,
    weighted_density_threshold_stds,
    num_runs,
    save_path="density_comparison_mean_variance.png"
)

plot_graphs.plot_mse_metrics(
    k_values,
    mse_density_naeem,
    mse_weighted_density,
    mse_weighted_density_threshold,
    save_path="MSEs_for_densities.png"
)

plot_graphs.plot_coverage_metrics(
    k_values,
    coverage_means,
    coverage_stds,
    weighted_coverage_means,
    weighted_coverage_stds,
    num_runs,
    coverage_save_path="coverage_naeem_mean_variance.png",
    weighted_coverage_save_path="weighted_coverage_mean_variance.png"
)