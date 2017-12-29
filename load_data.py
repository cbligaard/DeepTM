###### Loading data ######

# Libraries
import numpy as np
import pandas as pd


# Loading data and returning training, validation and test sets
def data_loading():

    #data = np.load('/zhome/e0/2/88697/deep_learning/data_tm.npz')
    data = np.load('data_tm.npz')
    
    X = data['X'] # Input protein encoding BLOSUM62, size (6733,2000,23)
    y = data['y'] # Protein label, size (6733,2000). Labels. 0: Membrane in-out, 1: Membrane out-in, 2: signal peptide, 3: inside, 4: outside.
    mask = data['mask']  # Sequence mask, you will need this to calculate the loss, as you want to mask the padding for proteins smaller than the maximum length, size (6733,2000).
    len_prot = data['len_prot'] # Length of each protein, size (6733)
    prot_type = data['prot_type'] # Protein type, size (6733). Labels. 0: Signal peptide, 1: Signal peptide with transmembrane, 2: Transmembrane, 3: Globular. 
    fold = data['fold'] # Partition assigned to each protein from 0 to 4, size (6733)
    ids = data['ids'] # Protein ids, size (6733).
    
    # Getting into Pandas format
    Xzipped = list(zip(X,y,len_prot,fold,prot_type,mask))
    df = pd.DataFrame(Xzipped, columns = ['X', 'y', 'len_prot', 'fold', 'prot_type', 'mask'])
    
    # Training set
    train_set = df[df.fold <= 2]
    
    # Validation set
    val_set = df[df.fold == 3]
    
    # Test set
    test_set = df[df.fold == 4]

    
    return(train_set, val_set, test_set)
