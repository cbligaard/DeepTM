##### Batch Generator

### Load library
import numpy as np
import pandas as pd
import math


# This function is made to generate mini batches for training.
# First, the proteins are sorted by length and mini-batches are made by putting proteins with similar length in the same batch. It is also ensured that there is same ratio of positive and negative data in each batch.
# In the end, we trim positions beyond the length of the longest protein in the batch to get more efficient training and finally, we shuffle order in the mini-batch.


### Function that creates mini batches with the correct requirements

def create_mini_batches(input_data, batch_size):
    # Sorting the dataframe based on length
    sorted_df = input_data.sort_values('len_prot')
    
    # Dividing the data in positive and negative examples (TM or not)
    positive = sorted_df[(sorted_df.prot_type == 1) | (sorted_df.prot_type == 2)]
    negative = sorted_df[(sorted_df.prot_type == 0) | (sorted_df.prot_type == 3)]
    
    # Getting the number of positive and negative
    n_positive = len(positive)
    n_negative = len(negative)
    
    # Calculating the number of batches
    n_batches = math.ceil(len(input_data) / batch_size)
    
    # Calculating the number of positive and negative in each batch
    pos_in_batch = n_positive / n_batches
    neg_in_batch = n_negative / n_batches   
    
    # Splitting the data into n_batches
    batches_pos = []
    batches_neg = []

    for group, df in positive.groupby(np.arange(len(positive)) // pos_in_batch):
        batches_pos.append(df)

    for group, df in negative.groupby(np.arange(len(negative)) // neg_in_batch):
        batches_neg.append(df)
        
    # Initiating the result arrays
    batches_X, batches_y, batches_mask, batches_y_type, batches_len = [], [], [], [], []
    
    # Getting the minibatches
    for i in range(n_batches):
        minibatch = pd.concat([batches_pos[i], batches_neg[i]])
        
        # Shuffle the minibatch
        shuffled = minibatch.sample(frac=1)
        
        # Get the length of longest protein
        length = minibatch.loc[minibatch['len_prot'].idxmax()][2]
        
        # Change to numpy array 
        X = pd.DataFrame.as_matrix(shuffled.X)
        y = pd.DataFrame.as_matrix(shuffled.y)
        mask = pd.DataFrame.as_matrix(shuffled['mask'])
        y_type = pd.DataFrame.as_matrix(shuffled['prot_type'])
        lens = pd.DataFrame.as_matrix(shuffled['len_prot'])
    
        # Trim X, y and mask based on length
        X_trimmed = np.array([i[:length] for i in X])
        y_trimmed = np.array([i[:length] for i in y])
        mask_trimmed = np.array([i[:length] for i in mask])
                
        # Append to the result
        batches_X.append(X_trimmed)
        batches_y.append(y_trimmed)
        batches_mask.append(mask_trimmed)  
        batches_y_type.append(np.asarray(y_type))
        batches_len.append(np.asarray(lens))
    
    # Return minibatches in an array of numpy arrays (X and corresponding y), the mask, y_type, and all true lengths
    return batches_X, batches_y, batches_mask, batches_y_type, batches_len
