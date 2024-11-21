import numpy as np
from scipy.special import expit 
import pandas as pd
import matplotlib.pyplot as plt
from prdc import compute_prdc
from scipy.stats import wishart
import plot_graphs
from generate_dataset import generate_real, generate_fake
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing  
from multiprocessing import Pool
import compute_metrics

# 1. Générer des variables explicatives 
seed = 41
np.random.seed(seed)





start_time = time.time()





m = 100
n = 10
num_runs  = 1
num_runs_outer = 10


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

    iter_start = time.time()


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

        iter_end = time.time()
        print(f"Iteration {iter + 1} took {iter_end - iter_start:.2f} seconds")
        print(f"Dataset number {iter + 1 } run {i + 1} completed.\n")


total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")















# Call the function to compute metrics
(density_naeem_means, density_naeem_stds, weighted_density_means, weighted_density_stds, 
 weighted_density_threshold_means, weighted_density_threshold_stds, mse_density_naeem, 
 mse_weighted_density, mse_weighted_density_threshold, coverage_means, coverage_stds, 
 weighted_coverage_means, weighted_coverage_stds) = compute_metrics.compute_metrics(density_Naeem_all, weighted_density_all, 
                                                                    coverage_all, weighted_coverage_all, 
                                                                    weighted_density_threshold_all, k_values)





metrics_dict = {
    'k': k_values,
    'density_naeem_mean': density_naeem_means,
    'density_naeem_std': density_naeem_stds,
    'weighted_density_mean': weighted_density_means,
    'weighted_density_std': weighted_density_stds,
    'weighted_density_threshold_mean': weighted_density_threshold_means,
    'weighted_density_threshold_std': weighted_density_threshold_stds,
    'coverage_mean': coverage_means,
    'coverage_std': coverage_stds,
    'weighted_coverage_mean': weighted_coverage_means,
    'weighted_coverage_std': weighted_coverage_stds,
    'mse_weighted_density': mse_weighted_density,
    'mse_weighted_density_threshold': mse_weighted_density_threshold
}

# Create DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Save with parameters in filename
filename = f'generative-evaluation-prdc/data/metrics_m{m}_n{n}_runs{num_runs}_lowrank{lowrank}.csv'
metrics_df.to_csv(filename, index=False)








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
    save_path=f"generative-evaluation-prdc/figures/MSEs_for_densities_across_{num_runs}_Run(s)_n_=_{n},_m_=_{m},_for_{num_runs_outer}_datasets.png"
)

plot_graphs.plot_coverage_metrics(
    k_values,
    coverage_means,
    coverage_stds,
    weighted_coverage_means,
    weighted_coverage_stds,
    num_runs,
    coverage_save_path=f"generative-evaluation-prdc/figures/coverage_naeem_mean_variance_across_{num_runs}_Run(s)_n_=_{n},_m_=_{m},_for_{num_runs_outer}_datasets.png",
    weighted_coverage_save_path=f"generative-evaluation-prdc/figures/weighted_coverage_mean_variance_across_{num_runs}_Run(s)_n_=_{n},_m_=_{m},_for_{num_runs_outer}_datasets.png"
)