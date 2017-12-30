##### Topology prediction

### Loading libraries
import numpy as np
import pandas as pd

### Function to find TM domains
def tm_count(protein):
    sets = []
    i = 0
    
    # Looping over protein length
    while i < len(protein):
        
        # Getting TM stretches
        if (int(protein[i]) == 0) or (int(protein[i]) == 1):
            tm_start = i
            j = i
            tm_type = int(protein[i])

            while (i < len(protein)) and (int(protein[i]) == tm_type):
                j += 1
                i += 1

            tm_end = j
            sets.append(set(range(tm_start, tm_end)))

        i += 1
    
    return(sets)


def topology_prediction(true_seq, pred_seq, true_type, pred_type):
     
    # Counting distribution of true classes and setting up count dictionary
    true_count = np.bincount(true_type)
    type_mat = {'SP+Glob': 0, 'SP+TM': [0,0], 'TM': [0,0], 'Glob': 0}
    
    
    # Looping over all proteins and checking topology
    for p in range(len(true_seq)):
        true = true_seq[p]
        pred = pred_seq[p]
        
        type_true = true_type[p]
        type_pred = pred_type[p]
    
        
        # Get SP + Globular (class 0) that are predicted to be of the right class
        if ((type_pred == 0) & (type_true == 0)):
            type_mat['SP+Glob'] += 1
        
        # Get Globular proteins (class 3) correctly predicted
        elif ((type_pred == 3) & (type_true == 3)):
            type_mat['Glob'] += 1
        
        
        # Checking topology for TM (class 2) and SP+TM (class 1) proteins predicted to be so
        elif (((type_pred == 1) & (type_true == 1)) | ((type_pred == 2) & (type_true == 2))):
    
            # Variable for topology check (1 if correct, 0 if wrong)
            tm_top = 1
    
            # Checking location of N and C termini
            if ((true[0] == pred[0]) & (true[-1] == pred[-1])):
    
                # Counting helices
                true_tm = tm_count(true)
                pred_tm = tm_count(pred)
    
                # Check if the number is the same
                if len(true_tm) == len(pred_tm):
    
                    # Checking if the true and predicted helices overlap
                    for h in range(len(true_tm)):
                        overlap = len(true_tm[h].intersection(pred_tm[h]))
                        if overlap < 5:
                            tm_top = 0
    
                else: 
                    tm_top = 0
    
            else:
                tm_top = 0
    
    
            # Saving result
            if type_pred == 1:
                type_mat['SP+TM'][tm_top] += 1
            else:
                type_mat['TM'][tm_top] += 1
    
            
    globsp = type_mat['SP+Glob'] / true_count[0] * 100
    glob = type_mat['Glob'] / true_count[3] * 100
    tm_corr = type_mat['TM'][1] / true_count[2] * 100
    tm_wrong = type_mat['TM'][0] / true_count[2] * 100
    tmsp_corr = type_mat['SP+TM'][1] / true_count[1] * 100
    tmsp_wrong = type_mat['SP+TM'][0] / true_count[1] * 100
    
    print('Topology prediction results (in percent):')
    print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('', 'TM', 'SP+TM', 'SP+Glob', 'Glob'))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}'.format('Correct', tm_corr, tmsp_corr, globsp, glob))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10} {:<10}'.format('Wrong top', tm_wrong, tmsp_wrong, '-', '-'))
    
    # Class 2: TM: No signal peptide, correct N + C, correct TM number + positions
    # Class 1: SP + TM: Signal peptide, correct N + C, correct TM number + positions
    # Class 0: SP + Globular: Signal peptide, no transmembrane
    # Class 3: Globular: No signal peptide, no transmembrane
    