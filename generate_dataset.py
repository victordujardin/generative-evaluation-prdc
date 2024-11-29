# data_generation.py

import numpy as np
import pandas as pd
from scipy.special import expit
import random 
import Jones

def generate_real(m=1000, lowrank=10, epsilon  = 0.0000000001):
    # np.random.seed(seed)
    X = np.random.multivariate_normal(np.zeros(lowrank), np.identity(lowrank), size=m)
    X = pd.DataFrame(X, columns=[f'X_{j}' for j in range(lowrank)])

    # Define coefficients
    beta = np.array([5.0] * lowrank)
    p = expit(np.dot(X, beta))
    w = 1 / p

    # Data augmentation with additional columns
    data_multi = X.copy()
    data_multi['mean_X'] = data_multi.mean(axis=1)
    # matrice_aleatoire = np.random.randn(lowrank, 32)
    # covariance_matrix = np.dot(np.dot(matrice_aleatoire.T, np.cov(X, rowvar=False)), matrice_aleatoire)

    # Generate additional distributions
    # n_pois = 10
    # data_multi_poisson = np.random.poisson(lam=abs(data_multi.iloc[:, random.randint(0, lowrank-1)]).values[:, np.newaxis], size=(m, n_pois))
    # poisson_columns = [f'poisson_{j}' for j in range(n_pois)]
    # data_multi[poisson_columns] = data_multi_poisson


    # multinomial_probs = expit(X.values)
    # multinomial_probs /= multinomial_probs.sum(axis=1, keepdims=True)
    # multinomial_samples = np.array([np.random.multinomial(1, p) for p in multinomial_probs])
    # multinomial_columns = [f'multinomial_{j}' for j in range(X.shape[1])]
    # data_multi[multinomial_columns] = multinomial_samples

    # mean_observations = np.dot(X, matrice_aleatoire)
    # multivariate_normal_samples = np.random.multivariate_normal(
    #     mean=np.zeros(mean_observations.shape[1]), cov=covariance_matrix, size=m
    # ) + mean_observations
    # multivariate_columns = [f'multivariate_{j}' for j in range(multivariate_normal_samples.shape[1])]
    # data_multi[multivariate_columns] = multivariate_normal_samples

    # lambda_exp = np.exp(np.abs(np.dot(X, np.array([0.2, -0.1, 0.4, -0.3, 0.1]))))
    # exponential_samples = np.random.exponential(scale=lambda_exp)
    # data_multi['exponential'] = exponential_samples


    #Generate combined variables with noise
    combined_vars = []
    combined_vars.append(X.iloc[:, 0] + X.iloc[:, 1] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 2] + X.iloc[:, 3] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 4] + X.iloc[:, 1] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 0] + X.iloc[:, 3] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 0] + X.iloc[:, 1] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 0] + X.iloc[:, 2] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 3] + X.iloc[:, 4] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 1] + X.iloc[:, 4] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 2] + X.iloc[:, 3] + np.random.normal(0, epsilon, size=m))
    combined_vars.append(X.iloc[:, 3] + X.iloc[:, 0] + np.random.normal(0, epsilon, size=m))
    combined_vars_df = pd.DataFrame(combined_vars).T
    combined_vars_df.columns = [f'combined_{i}' for i in range(1, len(combined_vars) + 1)]

    # print(Jones.jones(y_true = X, y_pred = combined_vars_df))

    for i, combined_var in enumerate(combined_vars, 1):
        data_multi[f'combined_{i}'] = combined_var




    X['w'] = w
    X['p'] = p
    data_multi= data_multi.drop(columns=['mean_X'])

    data_multi['w'] = w
    data_multi['p'] = p
    return X, data_multi

def generate_fake(m=1000, lowrank=10, epsilon  = 0.0000000001):
    Y = np.random.multivariate_normal(np.zeros(lowrank), np.identity(lowrank), size=m)
    Y = pd.DataFrame(Y, columns=[f'Y_{j}' for j in range(lowrank)])

    beta = np.array([-5.0]* lowrank)
    p_star = expit(np.dot(Y, beta))
    w_star = 1 / p_star

    fake_multi = Y.copy()
    # fake_multi['mean_Y'] = fake_multi.mean(axis=1)

    # matrice_aleatoire_fake = np.random.randn(lowrank, 32)
    # covariance_matrix_fake = np.dot(np.dot(matrice_aleatoire_fake.T, np.cov(Y, rowvar=False)), matrice_aleatoire_fake)

    # n_pois = 10
    # fake_multi_poisson = np.random.poisson(lam=abs(fake_multi.iloc[:, random.randint(0, lowrank-1)]).values[:, np.newaxis], size=(m, n_pois))
    # poisson_fake_columns = [f'poisson_fake_{j}' for j in range(n_pois)]
    # fake_multi[poisson_fake_columns] = fake_multi_poisson

    # multinomial_probs_fake = expit(Y.values)
    # multinomial_probs_fake /= multinomial_probs_fake.sum(axis=1, keepdims=True)
    # multinomial_samples_fake = np.array([np.random.multinomial(1, p) for p in multinomial_probs_fake])
    # multinomial_fake_columns = [f'multinomial_fake_{j}' for j in range(Y.shape[1])]
    # fake_multi[multinomial_fake_columns] = multinomial_samples_fake

    # mean_observations = np.dot(Y, matrice_aleatoire_fake)
    # multivariate_normal_samples = np.random.multivariate_normal(
    #     mean=np.zeros(mean_observations.shape[1]), cov=covariance_matrix_fake, size=m
    # ) + mean_observations
    # multivariate_columns = [f'multivariate_{j}' for j in range(multivariate_normal_samples.shape[1])]
    # fake_multi[multivariate_columns] = multivariate_normal_samples

    # lambda_exp_fake = np.exp(np.abs(np.dot(Y, np.array([0.2, -0.1, 0.4, -0.3, 0.1]))))
    # exponential_samples_fake = np.random.exponential(scale=lambda_exp_fake)
    # fake_multi['exponential_fake'] = exponential_samples_fake


    # Generate combined variables with noise
    combined_vars_fake = []
    combined_vars_fake.append(Y.iloc[:, 0] + Y.iloc[:, 1] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 2] + Y.iloc[:, 3] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 4] + Y.iloc[:, 1] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 0] + Y.iloc[:, 3] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 0] + Y.iloc[:, 1] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 0] + Y.iloc[:, 2] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 3] + Y.iloc[:, 4] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 1] + Y.iloc[:, 4] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 2] + Y.iloc[:, 3] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake.append(Y.iloc[:, 3] + Y.iloc[:, 0] + np.random.normal(0, epsilon, size=m))
    combined_vars_fake_df = pd.DataFrame(combined_vars_fake).T
    combined_vars_fake_df.columns = [f'combined_fake_{i}' for i in range(1, len(combined_vars_fake) + 1)]

    for i, combined_var_fake in enumerate(combined_vars_fake, 1):
        fake_multi[f'combined_fake_{i}'] = combined_var_fake



    Y['w_star'] = w_star
    Y['p_star'] = p_star

    fake_multi['w_star'] = w_star
    fake_multi['p_star'] = p_star
    return Y, fake_multi
