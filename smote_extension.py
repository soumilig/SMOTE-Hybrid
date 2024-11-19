import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors
import random

def populate(n, k, curr_samp_nn, df_samples):
    new_samps=[]
    while n!=0:
        nn = random.randint(0,k-1)
        print('Making',n,'sample')
    
        chosen_samp = df_samples[curr_samp_nn[nn]]
        curr_samp_vec = df_samples[curr_samp_nn[0]]
        diff = chosen_samp - curr_samp_vec
        gap = random.random()

        new_samp = list(curr_samp_vec + gap*diff)
        new_samps.append(new_samp)
        n=n-1

    return new_samps

## finds the cosine similiarity between two samples
def cosine_similiarity(sample1, sample2):
    term_num = np.dot(sample1, sample2)
    term_denom1 = np.linalg.norm(sample1)
    term_denom2 = np.linalg.norm(sample2)

    term_denom = term_denom1*term_denom2
    cos_sim = term_num/term_denom

    return cos_sim


def get_cos_vector(curr_sample, df_samples):
    distances = []
    for i in df_samples:
        distances.append(cosine_similiarity(curr_sample, i))
    
    sorted_indices = np.argsort(np.array(distances))[::-1]
    sorted_distances = np.array(distances)[sorted_indices]
    cos_vector_dict = {}
    for i in range(len(sorted_indices)):
        cos_vector_dict[sorted_indices[i]] = i

    sorted_cos_indices =dict(sorted(cos_vector_dict.items()))
    return sorted_cos_indices, sorted_distances, distances


## finds the euclidean distance between two samples
def euclidean_dist(sample1, sample2):

    terms = np.sum((sample1-sample2)**2)
    euc_dist = np.sqrt(terms)
    
    return euc_dist

def get_euc_vector(curr_sample, df_samples):
    distances = []
    for i in df_samples:
        distances.append(euclidean_dist(curr_sample, i))
    
    sorted_indices = np.argsort(np.array(distances))
    sorted_distances = np.array(distances)[sorted_indices]
    euc_vector_dict = {}
    for i in range(len(sorted_indices)):
        euc_vector_dict[sorted_indices[i]] = i
    
  
    sorted_euc_indices =dict(sorted(euc_vector_dict.items()))
    return sorted_euc_indices, sorted_distances, distances


def voting_mech(euc_dict, sorted_euc_dist, cosine_dict, sorted_cos_dist, euc_dist, cos_dist, k):
    add_dict = {}
    
    for i in euc_dict.keys():
        add_dict[i] = euc_dict[i]+cosine_dict[i]

    sorted_add_dict=dict(sorted(add_dict.items(), key=lambda item: item[1]))
    nn = list(sorted_add_dict.keys())
    nn = nn[0:k]
    nn_dist=[euc_dist[i] for i in nn]
    return nn, nn_dist


def smote(t, n, k, df, samples):
    """
    t -> number of minority class samples
    n -> amount of smote percent
    k -> number of nearest neighbors

    """
    df = df.drop(columns=['bug'])
    df_samples = np.array(df)
    if n<100:
        t = (n/100)*t
        n = 100

    n = int(n/100)
    num_attr = df.shape[1]
    newindex=0
    synthetic_samples = []
    total_samp2bgen = n*t
    nn_dict={}
    for i in range(t):
        cos_nn, sorted_cos_dist, cos_dist = get_cos_vector(samples[i], df_samples)
        euc_nn, sorted_euc_dist, euc_dist = get_euc_vector(samples[i], df_samples)
        nn, nn_dist = voting_mech(euc_nn, sorted_euc_dist, cos_nn, sorted_cos_dist, euc_dist, cos_dist, k)
        nn_dict[i] = (nn, nn_dist)
    
    std_dev_dict={}
    sum_std=0
    for i in nn_dict.keys():
        std_dev_dict[i]= [np.std(np.array(nn_dict[i][1]))]
        sum_std+=std_dev_dict[i][0] 
    
    for i in nn_dict.keys():
        x = std_dev_dict[i][0]
        std_dev_dict[i].append(x/sum_std)
        std_dev_dict[i].append(int(x/sum_std*total_samp2bgen))

    ## std_dev_dict={minor_sample_ind: [std_dev, proportion, number of samples to be generated]}
    
    for i in range(t):
        curr_samp = samples[i]
        nn = nn_dict[i][0]
        syn_sample = populate(std_dev_dict[i][2], k, nn, df_samples)
        for j in syn_sample:
            synthetic_samples.append(j)
            newindex+=1    
    
    synthetic_samples = np.array(synthetic_samples)
    return synthetic_samples



def get_minor_samples(df, minority_count):
    df_inter = pd.DataFrame(columns = df.columns)
    condition = df['bug']==1
    df_inter = df[condition]
    df_inter = df_inter.drop(columns=['bug'])
    minor_samples = np.array(df_inter)
    return minor_samples


def preprocess_df(df):
    df_new = df.drop(columns=['name', 'name.1', 'version'])
    bug_list = df_new['bug'].unique()
    for i in bug_list:
        if (i!=0):
            if (i!=1):
                df_new.replace(i,1, inplace=True)
        else:
            continue

    counts = df_new['bug'].value_counts()
    minority_count=counts[len(counts)-1]

    return df_new, minority_count


def main(df, smote_perc, k):
    df_2, minority_count = preprocess_df(df)
    minor_samples = get_minor_samples(df_2, minority_count)
    smote_samples = smote(minority_count, smote_perc, k, df_2, minor_samples)
    smote_samples = np.hstack((smote_samples, np.full((smote_samples.shape[0], 1), 1.0)))
    updated_df = pd.DataFrame(smote_samples, columns=df_2.columns)
    df_concat = pd.concat([df_2, updated_df], axis=0, ignore_index=True )
    ret_df = df_concat.sample(frac=1).reset_index(drop=True)
    return ret_df


if __name__=='__main__':
    main()
  