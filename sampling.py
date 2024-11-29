from generate_dataset import generate_real, generate_fake
import random
import pandas as pd

def generate_and_sampling_real(n, lowrank,epsilon = 0.0000000001):

    count = 0
    sampled_data = pd.DataFrame()
    while count <n:
        candidate, full_candidate = generate_real(1, lowrank, epsilon)
        if (random.random() < full_candidate['p'].iloc[0]):
            count += 1
            sampled_data = pd.concat([sampled_data, full_candidate], ignore_index=True)

    return sampled_data




def generate_and_sampling_fake(n, lowrank, epsilon = 0.0000000001):
    
        count = 0
        sampled_data = pd.DataFrame()
        while count <n:
            candidate, full_candidate = generate_fake(1, lowrank,epsilon)
            if (random.random() < full_candidate['p_star'].iloc[0]):
                count += 1
                sampled_data = pd.concat([sampled_data, full_candidate], ignore_index=True)
    
        return sampled_data
