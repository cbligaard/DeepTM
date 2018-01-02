### SCRIPT FOR TRAINING, VALIDATING AND SAVING THE DEEPTM MODEL
# Set-up made to run on DTU's HPC cluster (jobs submitted using the queuing system)

### Loading packages

import numpy as np
import tensorflow as tf
import os
import sys
import argparse


# Loading own functions
from minibatches import create_mini_batches
from plots import plots_performance
from conf_matrix import conf_matrix
from load_data import data_loading
from model import model_setup
from top_pred import topology_prediction
from confidence_plot import confidence_plot


### Reading command line arguments
### Argument parser
parser = argparse.ArgumentParser(description="Running neural network training")

parser.add_argument(
    "-learning_rate",
    required=False,
    type=float,
    dest="learning_rate",
    help='Learning rate to use in ADAM optimizer.')
parser.add_argument(
    "-CRF",
    required=False,
    action="store_true",
    help='Use CRF or not.')
parser.add_argument(
    "-n_hidden",
    required=False,
    type=int,
    dest="n_hidden",
    help='Number of hidden units in the LSTM cells.')
parser.add_argument(
    "-n_epochs",
    required=False,
    type=int,
    dest="n_epochs",
    help='Number of epochs to run per CV fold.')
parser.add_argument(
    "-batch_size",
    required=False,
    type=int,
    dest="batch_size",
    help='Batch size for training/validating.')

parser.set_defaults(n_epochs = 1000, batch_size = 64)

args = parser.parse_args()


### Getting a name for this setup to save the model and plots

name = 'lr_{0}_crf_{1}_batch_{2}_epochs_{3}_model'.format(args.learning_rate, args.CRF, args.batch_size, args.n_epochs)
n_hidden = 60
val_int = 10

### Loading data and setting hyper-parameters

# Loading the data
train_set, val_set, test_set = data_loading()

### Building the model

# Setting up the model
tf.reset_default_graph()
X_pl, mask_pl, y_pl, len_pl, y_type_pl, train_op, y_seq, y_seq_conf, y_type_pred, y_type_conf, loss, accuracy_seq, accuracy_type, transition_params = model_setup(learning_rate = args.learning_rate, n_hidden = n_hidden, CRF = args.CRF)

# Making saver object
saver = tf.train.Saver(save_relative_paths = True)

# Running through the data
print('Starting training with a learning rate of {0}, batch size {1}, {2} epochs, {3} units in the LSTMs and CRF = {4}.\n'.format(args.learning_rate, args.batch_size, args.n_epochs, n_hidden, args.CRF))
with tf.Session() as sess:
    try:
        # Reset the graph
        sess.run(tf.global_variables_initializer()) 
        
        # Reset best performance
        best_val_loss, best_epoch = np.inf, 0
        
        # Initializing lists to save performance metrics (names with 1 indicate predictions for amino acid locations and names with 2 indicate type-predictions)
        train_loss_all, val_loss_all, val_acc1_all, val_acc2_all = [], [], [], []
        test_loss_all, test_acc1_all, test_acc2_all = [], [], []
        val_true1_best, val_true2_best, val_pred1_best, val_pred2_best = [], [], [], []

        # Training set
        batches_X_train, batches_y_train, batches_mask_train, batches_y_type_train, \
        batches_len_train = create_mini_batches(train_set, args.batch_size)

        # Validation set 
        batches_X_val, batches_y_val, batches_mask_val, batches_y_type_val, \
        batches_len_val = create_mini_batches(val_set, args.batch_size)
        
        print("Epoch \tTrain loss \tVal loss \tVal acc 1 \tVal acc 2")

        # Starting the training loop
        for epoch in range(args.n_epochs):

            train_loss, val_loss, val_acc1, val_acc2, val_pred1, val_pred2 = [], [], [], [], [], [] 

            # Looping over minibatches and training on each one
            for batch in range(len(batches_X_train)):

                # Make fetches
                fetches_train = [train_op, loss] 

                # Set up feed dict (to feed the network with training data)
                feed_dict_train = {X_pl: batches_X_train[batch], mask_pl: batches_mask_train[batch], \
                                   y_pl: batches_y_train[batch], y_type_pl: batches_y_type_train[batch], \
                                   len_pl: batches_len_train[batch]}

                # Run the model and append the results to lists
                res_train = sess.run(fetches = fetches_train, feed_dict = feed_dict_train)
                
                train_loss.append(res_train[1])

                
            # Validate after running X rounds of training
            if (epoch+1) % val_int == 0:
                for batch in range(len(batches_X_val)):

                    # Run the model and append results
                    fetches_val = [y_seq, y_type_pred, loss, accuracy_seq, accuracy_type, transition_params]
                    
                    feed_dict_val = {X_pl: batches_X_val[batch], mask_pl: batches_mask_val[batch], \
                                     y_pl: batches_y_val[batch], y_type_pl: batches_y_type_val[batch], \
                                     len_pl: batches_len_val[batch]}
                    
                    res_val = sess.run(fetches = fetches_val, feed_dict = feed_dict_val)

                    val_pred1.extend([seq[batches_mask_val[batch][i].astype(bool)] for i, seq in enumerate(res_val[0])])
                    val_pred2.extend(res_val[1])
                    val_loss.append(res_val[2])
                    val_acc1.append(res_val[3])
                    val_acc2.append(res_val[4])

                # Printing status report of a full epoch and saving performance metrics for plotting
                train_loss_all.append(np.mean(train_loss))
                val_loss_all.append(np.mean(val_loss))
                val_acc1_all.append(np.mean(val_acc1))
                val_acc2_all.append(np.mean(val_acc2))

                print('{0} \t{1:.3f} \t\t{2:.3f} \t\t{3:.3f} \t\t{4:.3f}'.format(epoch+1, train_loss_all[-1],                                                                                val_loss_all[-1], val_acc1_all[-1],                                                                                val_acc2_all[-1]))

                # Saving the model variables if the validation loss decreased
                if val_loss_all[-1] < best_val_loss:
                    best_val_loss = val_loss_all[-1]
                    best_epoch = epoch
                    save_path = saver.save(sess, '/zhome/e0/2/88697/deep_learning/saved_models/{}/model'.format(name))

                    # Save prediction data for confusion matrices
                    val_true1 = []
                    val_true2 = []
                    for i in range(len(batches_y_val)):
                        val_true1.extend([batches_y_val[i][j][batches_mask_val[i][j].astype(bool)] \
                                          for j in range(len(batches_y_val[i]))])
                        val_true2.extend(batches_y_type_val[i])
                    val_pred1_best = val_pred1
                    val_pred2_best = val_pred2
                    
                    # Saving transition matrix for models using the CRF
                    if args.CRF:
                        best_transition_params = res_val[5]
                    

    except KeyboardInterrupt:
        pass

# Report best epoch
print('\nDone training. Model saved for epoch {0} saved in file: {1}. The validation loss was {2:.3f}.'.format(best_epoch+1, save_path, best_val_loss))


### Plots for performance
plots_performance(train_loss_all, val_loss_all, val_acc1_all, val_acc2_all, '/zhome/e0/2/88697/deep_learning/plots/' + name)

### Confusion matrices (best epoch)
conf_matrix(val_true1, val_pred1_best, val_true2, val_pred2_best)

### Topology prediction (best epoch)
topology_prediction(val_true1, val_pred1_best, val_true2, val_pred2_best)

### Transition matrix for CRF runs
if args.CRF:
    print('\nTransition matrix:')
    print(best_transition_params)

