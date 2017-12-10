##### Confusion matrices

# Import modules
import tensorflow as tf
import numpy as np

def conf_matrix(true_seq, pred_seq, true_type, pred_type):
    
    ### Confusion matrices for prediction 1 (position of each amino acid)
    # The actual class is in the rows and predicted class is in the columns.
    # We utilize data from the best epoch (lowest validation loss)

    conf_mat = tf.confusion_matrix(
    np.concatenate([i for batch in true_seq for i in batch]).ravel(),    # val_true is the true y for the entire validation set (best epoch)
    np.concatenate([i for batch in pred_seq for i in batch]).ravel(),    # val_pred is the prediction for the entire validation set (best epoch)
    num_classes = 5
    )
        
    with tf.Session():
        print('Prediction 1 (topolpgy) confusion matrix:')
        print(tf.Tensor.eval(conf_mat, feed_dict=None, session=None), '\n')
        
    ### Confusion matrices for prediction 2 (protein type)
    # The actual class is in the rows and predicted class is in the columns.
    # We utilize data from the best epoch (lowest validation loss)
    
    conf_mat = tf.confusion_matrix(
    np.concatenate([batch.flatten() for batch in true_type]).ravel().tolist(),    # val_true2 is the true y for the entire validation set (best epoch)
    np.concatenate([batch.flatten() for batch in pred_type]).ravel().tolist(),    # val_pred2 is the prediction for the entire validation set (best epoch)
    num_classes = 4
    )
    
    with tf.Session():
        print('Prediction 2 (type) confusion matrix:')
        print(tf.Tensor.eval(conf_mat, feed_dict=None, session=None), '\n')
