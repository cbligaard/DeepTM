##### Plotting

# Import modules
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# Plots
def plots_performance(train_loss, val_loss, val_acc, val_acc2, plot_name):
    
    # 1) Plot train and validation loss as a function of time
    fig = plt.figure(figsize=(16,12))
    epoch = np.arange(len(train_loss))
    
    fig.add_subplot(221)
    plt.title('Loss')
    plt.plot(epoch, train_loss, 'red', label='Train Loss')
    plt.plot(epoch, val_loss, 'blue', label='Validation Loss')
    plt.legend()
    plt.xlabel('Updates'), plt.ylabel('Loss')
    
    # 2) Plot train and validation accuracy as a function of time
    fig.add_subplot(222)
    plt.title('Accuracy')
    plt.plot(epoch, val_acc, 'blue', label='Validation Accuracy 1', linestyle=':')
    plt.plot(epoch, val_acc2, 'blue', label='Validation Accuracy 2', linestyle='--')
    plt.legend(loc=4)
    plt.xlabel('Updates'), plt.ylabel('Accuracy')
    
    plt.show()