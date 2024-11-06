import numpy as np
import sklearn.metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

__all__ = ['compute_prdc']


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


#does not work for the moment.
def get_kth_value_with_weight_threshold(unsorted, weights, threshold, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        weights: numpy.ndarray of the same shape as `unsorted`, representing weights associated with each value.
        threshold: float, the maximum allowed sum of weights.
        axis: int, the axis along which to perform the operation.
    Returns:
        kth values where the cumulative sum of the weights does not exceed the threshold.
    """
    # Sort the distances and get the indices of the sorted order
    sorted_indices = np.argsort(unsorted, axis=axis)
    sorted_values = np.take_along_axis(unsorted, sorted_indices, axis=axis)
    
    # Get the weights associated with the sorted distances (without sorting the weights themselves)
    associated_weights = np.take_along_axis(weights, sorted_indices, axis=axis)
    
    # Compute the cumulative sum of the associated weights along the specified axis
    cumulative_weights = np.cumsum(associated_weights, axis=axis)
    
    # Find the index where the cumulative sum of weights first exceeds the threshold
    # We need the last index where the cumulative weight is still less than or equal to the threshold
    valid_mask = cumulative_weights <= threshold
    valid_indices = np.where(valid_mask, np.arange(valid_mask.shape[axis]), -1)
    
    # Get the last valid index where the cumulative weight is within the threshold
    kth_indices = np.max(valid_indices, axis=axis)
    
    # Use the last valid index to get the k-th value from the sorted distances
    kth_values = np.take_along_axis(sorted_values, np.expand_dims(kth_indices, axis=axis), axis=axis).squeeze()

    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii




############ check that it works
def compute_nearest_neighbour_distances_threshold(input_features, nearest_k_weight, weights):
    distances = compute_pairwise_distance(input_features)
    weights_matrix = np.repeat(weights[np.newaxis, :], distances.shape[0], axis=0)
    radii = get_kth_value_with_weight_threshold(distances,weights=weights_matrix, threshold= nearest_k_weight + 1, axis = -1)
    return radii





def compute_prdc(real_features, fake_features, nearest_k, weights = None, weights_star = None, normalized = False):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    # real_nearest_neighbour_distances_with_threshold = compute_nearest_neighbour_distances_threshold(
    #     real_features, nearest_k, weights=weights)
    # fake_nearest_neighbour_distances_with_threshold = compute_nearest_neighbour_distances_threshold(
    #     fake_features, nearest_k, weights = weights_star)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)
    distance_real_real = compute_pairwise_distance(
        real_features, real_features)
    
    if weights is None:
        weights = np.ones(real_features.shape[0], dtype=np.float32)


    if weights_star is None:
        weights_star = np.ones(fake_features.shape[0], dtype=np.float32) 


    if not normalized: 

        weights_star = weights_star/ weights_star.sum()

        weights = weights  / weights.sum()








    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()
    


    density_Naeem = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()




    density_Hugues = (1. / float(nearest_k)) * (
        (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)) * np.expand_dims(weights/weights.sum(), axis=1)
    ).sum(axis=0).sum()


    
    
    indicator_fake = np.where(distance_real_fake <= np.expand_dims(real_nearest_neighbour_distances, axis=1), 1, 0)
    indicator_true = np.where(distance_real_real < np.expand_dims(real_nearest_neighbour_distances, axis=1), 1, 0)


    # Vectorized density update calculation
    numerator = (np.expand_dims(weights_star, axis=1) * indicator_fake.T).sum()
    denominator = (np.expand_dims(weights, axis=1) * indicator_true.T).sum()

    density_update2 = (numerator / denominator) 






    # indicator_fake_thresh = np.where(distance_real_fake <= np.expand_dims(real_nearest_neighbour_distances_with_threshold, axis=1), 1, 0)
    # indicator_true_thresh = np.where(distance_real_real < np.expand_dims(real_nearest_neighbour_distances_with_threshold, axis=1), 1, 0)


    # # Vectorized density update calculation
    # numerator_thresh = (np.expand_dims(weights_star, axis=1) * indicator_fake_thresh.T).sum()
    # denominator_thresh = (np.expand_dims(weights, axis=1) * indicator_true_thresh.T).sum()

    # density_update3 = (numerator_thresh / denominator_thresh) 






    
    



  
  

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()



    # Create a mask for distances within the ball of radius distance to the k nearest neighbor of x_i
    within_ball_mask = distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    # Create a mask for minimal distances
    minimal_distance_mask = distance_real_fake == np.expand_dims(distance_real_fake.min(axis=1), axis=1)
    # Combine both masks
    combined_mask = within_ball_mask & minimal_distance_mask

    #converting to numpy array
    weights = np.array(weights)
    weights_star = np.array(weights_star)

    # Perform the operation
    coverage_numerator = (weights[:, np.newaxis] * combined_mask).sum()
    # # Calculate the denominator
    coverage_denominator = weights.sum()
    # Calculate the coverage
    weighted_coverage = coverage_numerator  / coverage_denominator




    return dict(precision=precision, recall=recall,
                density_Hugues=density_Hugues, coverage=coverage, density_Naeem = density_Naeem, weighted_density = density_update2, weighted_coverage = weighted_coverage)




