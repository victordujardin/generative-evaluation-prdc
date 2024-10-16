from prdc import compute_prdc
import numpy as np
from sklearn.linear_model import LinearRegression


def simulations(n_simulations):
    for iteration in range(n_simulations):
        print("init iteration number", iteration + 1)

        # Simulate some data for testing
        np.random.seed(iteration)
        real_features = np.random.normal(loc=0.0, scale=1.0, size=(100, 128))  # 100 samples with 128-dimensional features
        fake_features = np.random.normal(loc=0.0, scale=1.0 + iteration, size=(100, 128))  # 100 samples with 128-dimensional features

        weights = np.random.uniform(low=1.0, high=100, size=(100))

        inverse_weights = 1 / weights


        # Fit linear regression model to explain the weights of real_features by the 128 features
        lin_reg = LinearRegression()
        lin_reg.fit(real_features, inverse_weights)

        # Predict weights for the observations of fake_features dataframe
        inverse_weights_star = lin_reg.predict(fake_features)


        weights_star = 1 / inverse_weights_star

        # Set the number of nearest neighbors to use
        nearest_k = 5

        # Compute PRDC metrics
        metrics = compute_prdc(real_features, fake_features, nearest_k, population_size=100, sample_size=100, weights= weights, weights_star=weights_star)

        precision = metrics['precision']
        recall = metrics['recall']
        density_Hugues = metrics['density_Hugues']
        coverage = metrics['coverage']
        density_Naeem = metrics['density_Naeem']
        weighted_density = metrics['weighted_density']
        weighted_coverage = metrics['weighted_coverage']

        print(f"Iteration {iteration + 1}:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Density Hugues: {density_Hugues}")
        print(f"Density Naeem: {density_Naeem}")
        print(f"Weighted Density: {weighted_density}")
        print(f"Coverage: {coverage}")
        print(f"Weighted Coverage: {weighted_coverage}")
        print("-" * 30)



simulations(n_simulations=10)