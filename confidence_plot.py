##### Plotting topology prediction and confidence per positions

# Import modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools

# Confidence for topology prediction
def confidence_plot(seq_pred, seq_conf, type_pred, type_conf, prot_len):
    # Translation table for types
    type_mat = {0:'Globular + Signal Peptide', 1:'Signal Peptide + Transmembrane', 2:'Transmembrane', 3:'Globular'}
    
    # Plot definitions
    plot_type = {0: [126, 'red', 20, '-'], 1: [126, 'red', 20, '-'], 2: [130, 'purple', 5, '--'], 3: [122.5, 'blue', 5, '-'], 4: [130, 'orange', 5, '-']}
    
    # Confidence for type
    print('The predicted type of this protein is "{0}". The confidence for the type prediction is {1:.2f} %.'.format(type_mat[type_pred], type_conf*100))
    
    # Topology plot
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_subplot(111)
    
    plt.plot(range(1, prot_len+1), seq_conf[0:prot_len]*100, color='k')
    plt.xlim(1, prot_len)
    plt.ylim(0,140)
    plt.xlabel('Amino acid number', fontsize=18)
    plt.ylabel('Confidence (%)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    start = 1
    for k, g in itertools.groupby(seq_pred):
        s = plot_type[k]
        stop = len(list(g))
        line = Line2D([start, start+stop], [s[0],s[0]], color = s[1], linewidth = s[2], linestyle = s[3], solid_capstyle="butt")
        ax.add_line(line)
        start += stop
    

