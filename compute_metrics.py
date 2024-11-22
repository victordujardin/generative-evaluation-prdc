
import numpy as np



def compute_metrics(density_Naeem_all, weighted_density_all, coverage_all, weighted_coverage_all, weighted_density_threshold_all, k_values):
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

    return (density_naeem_means, density_naeem_stds, weighted_density_means, weighted_density_stds, 
            weighted_density_threshold_means, weighted_density_threshold_stds, mse_density_naeem, 
            mse_weighted_density, mse_weighted_density_threshold, coverage_means, coverage_stds, 
            weighted_coverage_means, weighted_coverage_stds)
