import numpy as np
from scipy.special import expit 
import pandas as pd
import matplotlib.pyplot as plt
from prdc import compute_prdc
from scipy.stats import wishart
import plot_graphs
import compute_metrics
from generate_dataset import generate_real, generate_fake
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing  
from multiprocessing import Pool
import sampling
import Jones

def run_simulation(iter, m, n, lowrank, K_lim, num_runs, epsilon, addvar,beta, beta_star):
    density_Naeem_all = {k: [] for k in range(1, K_lim)}
    weighted_density_all = {k: [] for k in range(1, K_lim)}
    weighted_density_threshold_all = {k: [] for k in range(1, K_lim)}
    coverage_all = {k: [] for k in range(1, K_lim)}
    weighted_coverage_all = {k: [] for k in range(1, K_lim)}
    jones_list = []
    
    # real_X, data_multi = generate_real(m=m, lowrank=lowrank)
    # sampled_data = data_multi.sample(n=n, weights='p')


    jones, sampled_data = sampling.generate_and_sampling_real(n, lowrank, beta, addvar, epsilon)
    jones_list.append(jones)
    print(sampled_data)

    
    for i in range(num_runs):
        # Y, fake_data = generate_fake(m=m, lowrank=lowrank)

        # sampled_data_fake = fake_data.sample(n=n, weights='p_star')


        sampled_data_fake = sampling.generate_and_sampling_fake(n, lowrank, beta_star,addvar ,epsilon)
        print(sampled_data_fake)


        
        for k in range(1, K_lim):
            metrics = compute_prdc(
                sampled_data.drop(columns=['p', 'w']),
                sampled_data_fake.drop(columns=['p_star', 'w_star']),
                k,
                weights=sampled_data["w"],
                weights_star=sampled_data_fake["w_star"],
                normalized=False,
                weight_threshold=k * ((sampled_data["w"] / sampled_data["w"].sum()).mean())
            )
            
            density_Naeem_all[k].append(metrics.get('density_Naeem', np.nan))
            weighted_density_all[k].append(metrics.get('weighted_density', np.nan))
            coverage_all[k].append(metrics.get('coverage', np.nan))
            weighted_coverage_all[k].append(metrics.get('weighted_coverage', np.nan))
            weighted_density_threshold_all[k].append(metrics.get('weighted_density_threshold', np.nan))
    
    print(f"Iteration {iter + 1} completed.")
    return (density_Naeem_all, weighted_density_all, coverage_all, weighted_coverage_all, weighted_density_threshold_all, jones_list)

if __name__ == "__main__":
    seed = 41
    np.random.seed(seed)
    start_time = time.time()
    m = 250
    n = 400
    num_runs = 1
    num_runs_outer = 48
    epsilon = 0.1
    addvar = True
    lowrank = 5
    beta = np.array([-5.0, 5.0,5.0,5.0,-5.0])#np.array([5.0] * lowrank)
    beta_star = np.array([5.0, -5.0,-5.0,-5.0,5.0]) #np.array([-5.0] * lowrank)


    K_lim = n - 1
    k_values = list(range(1, K_lim))
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_simulation, iter, m, n, lowrank, K_lim, num_runs, epsilon, addvar, beta, beta_star)
            for iter in range(num_runs_outer)
        ]
        
        results = [future.result() for future in futures]
    
    density_Naeem_all = {k: [] for k in range(1, K_lim)}
    weighted_density_all = {k: [] for k in range(1, K_lim)}
    weighted_density_threshold_all = {k: [] for k in range(1, K_lim)}
    coverage_all = {k: [] for k in range(1, K_lim)}
    weighted_coverage_all = {k: [] for k in range(1, K_lim)}
    jones_list = []
    
    for res in results:
        for k in range(1, K_lim):
            density_Naeem_all[k].extend(res[0][k])
            weighted_density_all[k].extend(res[1][k])
            coverage_all[k].extend(res[2][k])
            weighted_coverage_all[k].extend(res[3][k])
            weighted_density_threshold_all[k].extend(res[4][k])
        jones_list.extend(res[5])

    if jones_list:
        avg_jones = np.mean(jones_list)
        print(f"Mean Jones coefficient: {avg_jones:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    (density_naeem_means, density_naeem_stds, weighted_density_means, weighted_density_stds, 
     weighted_density_threshold_means, weighted_density_threshold_stds, mse_density_naeem, 
     mse_weighted_density, mse_weighted_density_threshold, coverage_means, coverage_stds, 
     weighted_coverage_means, weighted_coverage_stds) = compute_metrics.compute_metrics(
        density_Naeem_all, weighted_density_all, coverage_all, weighted_coverage_all, 
        weighted_density_threshold_all, k_values
    )
    
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
    
    metrics_df = pd.DataFrame(metrics_dict)
    filename = f'prdc/data/metrics_m{m}_n{n}_runs{num_runs}_lowrank5.csv'
    metrics_df.to_csv(filename, index=False)
    
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
        5,
        save_path=f"prdc/figures/newdata_wstar_parallel_density_comparison_mean_variance_across_{num_runs}_Run(s)_n_{n}_for_{num_runs_outer}_datasets_jones_{avg_jones}.png"
    )
    
    plot_graphs.plot_mse_metrics(
        k_values,
        mse_density_naeem,
        mse_weighted_density,
        mse_weighted_density_threshold,
        save_path=f"prdc/figures/newdata_wstar_parallel_MSEs_for_densities_across_{num_runs}_Run(s)_n_{n}_for_{num_runs_outer}_datasets_jones_{avg_jones}.png"
    )
    
    plot_graphs.plot_coverage_metrics(
        k_values,
        coverage_means,
        coverage_stds,
        weighted_coverage_means,
        weighted_coverage_stds,
        num_runs,
        coverage_save_path=f"prdc/figures/coverage_naeem_mean_variance_across_{num_runs}_Run(s)_n_{n}_m_{m}_for_{num_runs_outer}_datasets_jones_{avg_jones}.png",
        weighted_coverage_save_path=f"prdc/figures/weighted_coverage_mean_variance_across_{num_runs}_Run(s)_n_{n}_m_{m}_for_{num_runs_outer}_datasets_jones_{avg_jones}.png"
    ) 
