##### Batch Generator

### Load library
import numpy as np
import pandas as pd
import math


# This function is made to generate mini batches for training.
# First, the proteins are sorted by length and mini-batches are made by putting proteins with similar length in the same batch. It is also ensured that there is same ratio of positive and negative data in each batch.
 
 
# In the end, we trim positions beyond the length of the longest protein in the batch to get more efficient training and finally, we shuffle order in the mini-batch.


### Function that creates mini batches with the correct requirements

def createMiniBatches(trainingDataFrame, batch_size):
    # Sorting the dataframe based on length
    sortedDF = trainingDataFrame.sort_values('len_prot')
    
    # Dividing the data in positive and negative
    positive = sortedDF[(sortedDF.prot_type == 1) | (sortedDF.prot_type == 2)]
    negative = sortedDF[(sortedDF.prot_type == 0) | (sortedDF.prot_type == 3)]
    
    # Getting the number of positive and negative
    noPositive = len(positive)
    noNegative = len(negative)
    
    # Calculating the number of batches
    noOfBatches = math.ceil(len(trainingDataFrame) / batch_size)
    
    # Calculating the number of positive and negative in each batch
    noPosInBatch = noPositive / noOfBatches
    noNegInBatch = noNegative / noOfBatches   
    
    # Splitting the data into noOfBacthes
    miniBatches_pos = []
    miniBatches_neg = []

    for group, df in positive.groupby(np.arange(len(positive)) // noPosInBatch):
        miniBatches_pos.append(df)

    for group, df in negative.groupby(np.arange(len(negative)) // noNegInBatch):
        miniBatches_neg.append(df)
        
    # Initiating the result arrays
    miniBatches_X, miniBatches_y, miniBatches_mask, miniBatches_yType, miniBatches_len = [], [], [], [], []
    
    # Getting the minibatches
    for i in range(noOfBatches):
        minibatch = pd.concat([miniBatches_pos[i], miniBatches_neg[i]])
        
        # Shuffle the minibatch
        shuffled = minibatch.sample(frac=1)
        
        # Get the length of longest protein
        length = minibatch.loc[minibatch['len_prot'].idxmax()][2]
        
        # Change to numpy array 
        X = pd.DataFrame.as_matrix(shuffled.X)
        y = pd.DataFrame.as_matrix(shuffled.y)
        mask = pd.DataFrame.as_matrix(shuffled['mask'])
        yType = pd.DataFrame.as_matrix(shuffled['y_type'])
        lens = pd.DataFrame.as_matrix(shuffled['len_prot'])
    
        # Trim X, y and mask based on length
        X_trimmed = np.array([i[:length] for i in X])
        y_trimmed = np.array([i[:length] for i in y])
        mask_trimmed = np.array([i[:length] for i in mask])
        
        #Changing yType's lists into numpy arrays
        yTypeNP = [np.asarray(i) for i in yType]
        
        # Append to the result
        miniBatches_X.append(X_trimmed)
        miniBatches_y.append(y_trimmed)
        miniBatches_mask.append(mask_trimmed)  
        miniBatches_yType.append(np.asarray(yTypeNP))
        miniBatches_len.append(np.asarray(lens))
    
    # Return minibatches in an array of numpy arrays (X and corresponding y), the mask, y_type, and all true lengths
    return miniBatches_X, miniBatches_y, miniBatches_mask, miniBatches_yType, miniBatches_len
