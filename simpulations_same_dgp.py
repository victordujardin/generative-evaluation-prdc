import numpy as np
from scipy.special import expit  # Fonction sigmoïde
import pandas as pd
import matplotlib.pyplot as plt
from prdc import compute_prdc
from scipy.stats import wishart
import plot_graphs
from generate_dataset import generate_real, generate_fake
import time



curr = time.time()



# 1. Générer des variables explicatives 
# seed = 41
# np.random.seed(seed)


m = 1000
n = 500
num_runs  = 1
num_runs_outer = 100


K_lim = n - 1
k_values = list(range(1, K_lim))
density_Naeem_all = {k: [] for k in range(1, K_lim)}
weighted_density_all = {k: [] for k in range(1, K_lim)}
weighted_density_threshold_all = {k: [] for k in range(1, K_lim)}
coverage_all = {k: [] for k in range(1, K_lim)}
weighted_coverage_all = {k: [] for k in range(1, K_lim)}

lowrank = 5





for iter in range(num_runs_outer):
    # dynamic_seed = seed + iter
    # np.random.seed(dynamic_seed)


    real_X, data_multi = generate_real(m=m, lowrank=lowrank)
    sampled_data = real_X.sample(n=n, weights='p') #, random_state=seed)

    



    for i in range(num_runs):

        # sub_seed = dynamic_seed + i 
        # np.random.seed(sub_seed)



        Y, fake_data = generate_fake(m=m, lowrank = lowrank)
        sampled_data_fake = Y.sample(n=n) #, weights='p_star')

        for k in range(1, K_lim):

            # Assume compute_prdc is a function that returns a dictionary with the required metrics
            metrics = compute_prdc(sampled_data.drop(columns=['p', 'w']), sampled_data_fake.drop(columns=['p_star', 'w_star']), k, weights=sampled_data["w"], weights_star=None, normalized = False, weight_threshold=k*((sampled_data["w"]/(sampled_data["w"].sum())).mean()))
            # metrics = compute_prdc(sampled_data.iloc[:, :-2], fakedata.iloc[: , :-2], k, weights=sampled_data["w"], weights_star=None, normalized = False, weight_threshold=k*((sampled_data["w"]/(sampled_data["w"].sum())).mean()))
            # metrics = compute_prdc(sampled_data.iloc[:, :-2], sampled_data_fake.iloc[: , :-2], k, weights=sampled_data["w"], weights_star=sampled_data_fake["w_star"], normalized = False, weight_threshold=k*((sampled_data["w"]/(sampled_data["w"].sum())).mean()))
            # metrics = compute_prdc(sampled_data.iloc[:, :-2], sampled_data_fake.iloc[: , :-2], k, weights=None, weights_star=None, normalized = False, weight_threshold=k /n)
            
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


        print(f"Dataset number {iter + 1 } run {i + 1} completed.\n")




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





# Prepare coverage metrics
coverage_means = [mean_coverage[k] for k in k_values]
coverage_vars = [var_coverage[k] for k in k_values]
coverage_stds = np.sqrt(coverage_vars)

weighted_coverage_means = [mean_weighted_coverage[k] for k in k_values]
weighted_coverage_vars = [var_weighted_coverage[k] for k in k_values]
weighted_coverage_stds = np.sqrt(weighted_coverage_vars)

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
    n, 
    m,
    num_runs_outer,
    lowrank,
    save_path=f"generative-evaluation-prdc/figures/density_comparison_mean_variance_across_{num_runs}_Run(s)_n_=_{n},_m_=_{m},_for_{num_runs_outer}_datasets.png"
)

plot_graphs.plot_mse_metrics(
    k_values,
    mse_density_naeem,
    mse_weighted_density,
    mse_weighted_density_threshold,
    save_path="generative-evaluation-prdc/figures/MSEs_for_densities.png"
)

plot_graphs.plot_coverage_metrics(
    k_values,
    coverage_means,
    coverage_stds,
    weighted_coverage_means,
    weighted_coverage_stds,
    num_runs,
    coverage_save_path="generative-evaluation-prdc/figures/coverage_naeem_mean_variance.png",
    weighted_coverage_save_path="generative-evaluation-prdc/figures/weighted_coverage_mean_variance.png"
)