##### Plotting topology prediction and confidence per positions

# Import modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import matplotlib.patches as patches
import itertools


# Confidence for topology prediction
def confidence_plot(seq_pred, seq_true, seq_conf, type_pred, type_true, type_conf, prot_len):
    # Translation table for types
    type_mat = {0:'Globular + Signal Peptide', 1:'Signal Peptide + Transmembrane', 2:'Transmembrane', 3:'Globular'}
    
    # Plot definitions
    plot_pred_type = {0: [126, 'red', 20, '-'], 1: [126, 'red', 20, '-'], 2: [129.5, 'purple', 5, '--'], 3: [123.5, 'blue', 5, '-'], 4: [129.5, 'orange', 5, '-']}
    plot_true_type = {0: [1.15, 'red', 20, '-'], 1: [1.15, 'red', 20, '-'], 2: [1.75, 'purple', 5, '--'], 3: [0.7, 'blue', 5, '-'], 4: [1.75, 'orange', 5, '-']}
    
    
    # Confidence for type
    print('The predicted type of this protein is "{0}". The confidence for the type prediction is {1:.2f} %.\nThe true type is "{2}".'.format(type_mat[type_pred], type_conf*100, type_mat[type_true]))
    
    
    ### Topology plot
    fig = plt.figure(figsize=(16,7))
    gs = gridspec.GridSpec(2, 2, height_ratios = [1, 12], width_ratios = [20, 1], hspace = 0.05, wspace = 0.06) 
    
    
    # Topology plot for true sequence
    ax = plt.subplot(gs[0])
    plt.plot()
    plt.axis('off')
    plt.xlim(1, prot_len)
    plt.ylim(0,2)
    
    start = 1
    for k, g in itertools.groupby(seq_pred):
        s = plot_true_type[k]
        stop = len(list(g))
        line = Line2D([start, start+stop], [s[0],s[0]], color = s[1], linewidth = s[2], linestyle = s[3], solid_capstyle="butt")
        ax.add_line(line)
        start += stop
    
    
    # Label
    ax = plt.subplot(gs[1])
    plt.plot()
    plt.axis('off')
    plt.xlim(1, 5)
    plt.ylim(0, 2)
    ax.text(0, 0.5, 'True sequence', fontsize=18)
        
    
    # Topology plot for predicted sequence
    ax = plt.subplot(gs[2])
    plt.plot(range(1, prot_len+1), seq_conf[0:prot_len]*100, color='k', )
    plt.xlim(1, prot_len)
    plt.ylim(0,140)
    plt.xlabel('Amino acid number', fontsize=18)
    plt.ylabel('Confidence (%)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.get_major_ticks()[-1].set_visible(False)
    ax.yaxis.get_major_ticks()[-2].set_visible(False)
    
    start = 1
    for k, g in itertools.groupby(seq_pred):
        s = plot_pred_type[k]
        stop = len(list(g))
        line = Line2D([start, start+stop], [s[0],s[0]], color = s[1], linewidth = s[2], linestyle = s[3], solid_capstyle="butt")
        ax.add_line(line)
        start += stop
        
    
    # Label
    ax = plt.subplot(gs[3])
    plt.plot()
    plt.axis('off')
    plt.xlim(1, 5)
    plt.ylim(0, 140)
    ax.text(0, 118, 'Predicted sequence\n{0:.2f} % accuracy'.format(sum(seq_pred == seq_true) / prot_len * 100), fontsize=18)
    
    plt.show()