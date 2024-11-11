import matplotlib.pyplot as plt
import numpy as np

def plot_density_metrics(k_values, density_naeem_means, density_naeem_stds,
                         weighted_density_means, weighted_density_stds,
                         weighted_density_threshold_means, weighted_density_threshold_stds,
                         num_runs, save_path="density_comparison_mean_variance.png"):
    plt.figure(figsize=(14, 10))
    
    # Plot Mean Density Naeem with variance
    plt.plot(k_values, density_naeem_means, label='Mean Density Naeem', marker='o', color='blue')
    plt.fill_between(k_values,
                     np.array(density_naeem_means) - density_naeem_stds,
                     np.array(density_naeem_means) + density_naeem_stds,
                     color='blue', alpha=0.2, label='Variance Density Naeem')
    
    # Plot Mean Weighted Density with variance
    plt.plot(k_values, weighted_density_means, label='Mean Weighted Density', marker='x', color='red')
    plt.fill_between(k_values,
                     np.array(weighted_density_means) - weighted_density_stds,
                     np.array(weighted_density_means) + weighted_density_stds,
                     color='red', alpha=0.2, label='Variance Weighted Density')
    
    # Plot Mean Weighted Density Threshold with variance
    plt.plot(k_values, weighted_density_threshold_means, label='Mean Weighted Density Threshold', marker='s', color='green')
    plt.fill_between(k_values,
                     np.array(weighted_density_threshold_means) - weighted_density_threshold_stds,
                     np.array(weighted_density_threshold_means) + weighted_density_threshold_stds,
                     color='green', alpha=0.2, label='Variance Weighted Density Threshold')
    
    # Customize the plot
    plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
    plt.ylabel('Density Metrics', fontsize=14)
    plt.title(f'Mean and Variance of Density Metrics across {num_runs} Run(s)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(k_values)  # Ensure all k values are marked on the x-axis
    
    # Save the combined plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()

def plot_mse_metrics(k_values, mse_density_naeem, mse_weighted_density, mse_weighted_density_threshold, save_path="MSEs_for_densities.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mse_density_naeem, marker='o', linestyle='-', label='MSE Density Naeem')
    plt.plot(k_values, mse_weighted_density, marker='s', linestyle='-', label='MSE Weighted Density')
    plt.plot(k_values, mse_weighted_density_threshold, marker='^', linestyle='-', label='MSE Weighted Density Threshold')
    
    # Adding titles and labels
    plt.title('Mean Squared Error (MSE) for Each Metric vs k')
    plt.xlabel('k')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_coverage_metrics(k_values, coverage_means, coverage_stds, 
                         weighted_coverage_means, weighted_coverage_stds,
                         num_runs,
                         coverage_save_path="coverage_naeem_mean_variance.png",
                         weighted_coverage_save_path="weighted_coverage_mean_variance.png"):
    # Plot Coverage Naeem
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, coverage_means, label='Mean Coverage Naeem', marker='o', color='g')
    plt.fill_between(k_values,
                     np.array(coverage_means) - coverage_stds,
                     np.array(coverage_means) + coverage_stds,
                     color='g', alpha=0.2, label='Variance Coverage Naeem')
    plt.xlabel('k')
    plt.ylabel('Coverage Naeem')
    plt.title(f'Mean and Variance of Coverage Naeem across {num_runs} Runs')
    plt.legend()
    plt.grid(True)
    plt.savefig(coverage_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Weighted Coverage
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, weighted_coverage_means, label='Mean Weighted Coverage', marker='x', color='m')
    plt.fill_between(k_values,
                     np.array(weighted_coverage_means) - weighted_coverage_stds,
                     np.array(weighted_coverage_means) + weighted_coverage_stds,
                     color='m', alpha=0.2, label='Variance Weighted Coverage')
    plt.xlabel('k')
    plt.ylabel('Weighted Coverage')
    plt.title(f'Mean and Variance of Weighted Coverage across {num_runs} Runs')
    plt.legend()
    plt.grid(True)
    plt.savefig(weighted_coverage_save_path, dpi=300, bbox_inches='tight')
    plt.show()
