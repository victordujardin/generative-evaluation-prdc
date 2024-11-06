import random
from typing import List, Any

def sequential_weighted_sample(
    data: List[Any],
    weights: List[float],
    n: int,
    seed: int = None
) -> List[Any]:
    """
    Perform sequential weighted random sampling without replacement.
    If a full pass does not yield enough samples, repeat the process until n samples are selected.

    Parameters:
    - data (List[Any]): The list of observations to sample from.
    - weights (List[float]): The list of selection probabilities for each observation.
                             Each probability should be in the range [0, 1].
    - n (int): The number of observations to sample.
    - seed (int, optional): Seed for the random number generator to ensure reproducibility.

    Returns:
    - List[Any]: A list of sampled observations.
    """

    if seed is not None:
        random.seed(seed)

    if len(data) != len(weights):
        raise ValueError("The length of data and weights must be the same.")

    if not 0 < n <= len(data):
        raise ValueError("Sample size 'n' must be between 1 and the number of observations in data.")

    # Pair each observation with its corresponding weight
    paired_data = list(zip(data, weights))

    # Shuffle the paired data to randomize the order
    random.shuffle(paired_data)

    sampled = []
    remaining = paired_data.copy()

    # To prevent infinite loops, track the number of passes
    max_passes = 100000000000000000000000000000000000  # Arbitrary large number; adjust as needed
    current_pass = 0
    iterations = 0

    while len(sampled) < n  : #and current_pass < max_passes:
        current_pass += 1
        print("current pass ", current_pass)
        # Iterate through a copy of remaining to allow modification during iteration
        for observation, weight in remaining.copy():
            if len(sampled) >= n:
                break  # Desired sample size achieved

            if not (0 <= weight <= 1):
                raise ValueError("All weights must be between 0 and 1.")

            r = random.random()
            if r <= weight:
                iterations+=1
                print(iterations)
                print(weight)
                sampled.append(observation)
                remaining.remove((observation, weight))  # Remove selected observation


        # If a full pass didn't yield enough, continue to the next pass
        if current_pass >= max_passes:
            raise RuntimeError("Maximum number of passes reached. Unable to select the desired number of samples.")

    if len(sampled) < n:
        raise ValueError("Not enough observations could be selected based on the given weights to meet the desired sample size.")

    return sampled

