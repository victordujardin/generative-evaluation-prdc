# data_generation.py

import numpy as np
import pandas as pd
from scipy.special import expit

def generate_real(m=1000, lowrank=10):
    # np.random.seed(seed)
    X = np.random.multivariate_normal(np.zeros(lowrank), np.identity(lowrank), size=m)
    X = pd.DataFrame(X, columns=[f'X_{j}' for j in range(lowrank)])

    # Define coefficients
    beta = np.array([5.0, -5.0, -5.0, -5.0, 5.0])
    p = expit(np.dot(X, beta))
    w = 1 / p

    # Data augmentation with additional columns
    data_multi = X.copy()
    data_multi['mean_X'] = data_multi.mean(axis=1)
    matrice_aleatoire = np.random.randn(lowrank, 32)
    covariance_matrix = np.dot(np.dot(matrice_aleatoire.T, np.cov(X, rowvar=False)), matrice_aleatoire)

    # Generate additional distributions
    n_pois = 10
    data_multi_poisson = np.random.poisson(lam=abs(data_multi['mean_X']).values[:, np.newaxis], size=(m, n_pois))
    poisson_columns = [f'poisson_{j}' for j in range(n_pois)]
    data_multi[poisson_columns] = data_multi_poisson

    multinomial_probs = expit(X.values)
    multinomial_probs /= multinomial_probs.sum(axis=1, keepdims=True)
    multinomial_samples = np.array([np.random.multinomial(1, p) for p in multinomial_probs])
    multinomial_columns = [f'multinomial_{j}' for j in range(X.shape[1])]
    data_multi[multinomial_columns] = multinomial_samples

    mean_observations = np.dot(X, matrice_aleatoire)
    multivariate_normal_samples = np.random.multivariate_normal(
        mean=np.zeros(mean_observations.shape[1]), cov=covariance_matrix, size=m
    ) + mean_observations
    multivariate_columns = [f'multivariate_{j}' for j in range(mean_observations.shape[1])]
    data_multi[multivariate_columns] = multivariate_normal_samples

    lambda_exp = np.exp(np.abs(np.dot(X, np.array([0.2, -0.1, 0.4, -0.3, 0.1]))))
    exponential_samples = np.random.exponential(scale=lambda_exp)
    data_multi['exponential'] = exponential_samples

    X['w'] = w
    X['p'] = p

    data_multi['w'] = w
    data_multi['p'] = p
    return X, data_multi

def generate_fake(m=1000, lowrank=10):
    Y = np.random.multivariate_normal(np.zeros(lowrank), np.identity(lowrank), size=m)
    Y = pd.DataFrame(Y, columns=[f'Y_{j}' for j in range(lowrank)])

    beta = np.array([5.0, -5.0, -5.0, -5.0, 5.0])
    p_star = expit(np.dot(Y, beta))
    w_star = 1 / p_star

    fake_multi = Y.copy()
    fake_multi['mean_Y'] = fake_multi.mean(axis=1)

    matrice_aleatoire_fake = np.random.randn(lowrank, 32)
    covariance_matrix_fake = np.dot(np.dot(matrice_aleatoire_fake.T, np.cov(Y, rowvar=False)), matrice_aleatoire_fake)

    n_pois = 10
    fake_multi_poisson = np.random.poisson(lam=abs(fake_multi['mean_Y']).values[:, np.newaxis], size=(m, n_pois))
    poisson_fake_columns = [f'poisson_fake_{j}' for j in range(n_pois)]
    fake_multi[poisson_fake_columns] = fake_multi_poisson

    multinomial_probs_fake = expit(Y.values)
    multinomial_probs_fake /= multinomial_probs_fake.sum(axis=1, keepdims=True)
    multinomial_samples_fake = np.array([np.random.multinomial(1, p) for p in multinomial_probs_fake])
    multinomial_fake_columns = [f'multinomial_fake_{j}' for j in range(Y.shape[1])]
    fake_multi[multinomial_fake_columns] = multinomial_samples_fake

    mean_observations_fake = np.dot(Y, matrice_aleatoire_fake)
    multivariate_normal_samples_fake = np.random.multivariate_normal(
        mean=np.zeros(mean_observations_fake.shape[1]), cov=covariance_matrix_fake, size=m
    ) + mean_observations_fake
    multivariate_fake_columns = [f'multivariate_fake_{j}' for j in range(mean_observations_fake.shape[1])]
    fake_multi[multivariate_fake_columns] = multivariate_normal_samples_fake

    lambda_exp_fake = np.exp(np.abs(np.dot(Y, np.array([0.2, -0.1, 0.4, -0.3, 0.1]))))
    exponential_samples_fake = np.random.exponential(scale=lambda_exp_fake)
    fake_multi['exponential_fake'] = exponential_samples_fake



    Y['w_star'] = w_star
    Y['p_star'] = p_star

    fake_multi['w_star'] = w_star
    fake_multi['p_star'] = p_star
    return Y, fake_multi
