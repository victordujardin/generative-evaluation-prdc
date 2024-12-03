from generate_dataset import generate_real, generate_fake
import random
import pandas as pd
import Jones
import numpy as np
def generate_and_sampling_real(n, lowrank,beta,addvar, epsilon):

    count = 0
    sampled_data = pd.DataFrame()
    while count <n:
        candidate, full_candidate = generate_real(1, lowrank, beta,epsilon, addvar)
        if (random.random() < full_candidate['p'].iloc[0]):
            count += 1
            sampled_data = pd.concat([sampled_data, full_candidate], ignore_index=True)
    lowrank_data = sampled_data.filter(like='X')
    combined_data = sampled_data.filter(like='combined')
    if lowrank_data.shape[1] > 0 and combined_data.shape[1] > 0:
        jones_coefficient = Jones.jones(np.asarray(lowrank_data), np.asarray(combined_data))
        print("Jones coefficient between lowrank and combined variables:", jones_coefficient)
    return jones_coefficient, sampled_data




def generate_and_sampling_fake(n, lowrank, beta,addvar,epsilon):
    
        count = 0
        sampled_data = pd.DataFrame()
        while count <n:
            candidate, full_candidate = generate_fake(1, lowrank,beta,epsilon, addvar)
            if (random.random() < full_candidate['p_star'].iloc[0]):
                count += 1
                sampled_data = pd.concat([sampled_data, full_candidate], ignore_index=True)
    
        return sampled_data
