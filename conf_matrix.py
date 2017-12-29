##### Confusion matrices

# Import modules
import tensorflow as tf
import numpy as np

def conf_matrix(true_seq, pred_seq, true_type, pred_type):
    
    ### Confusion matrices for prediction 1 (position of each amino acid)
    # The actual class is in the rows and predicted class is in the columns.
    # We utilize data from the best epoch (lowest validation loss)

    conf_mat = tf.confusion_matrix(
    np.concatenate(true_seq),    # val_true is the true y for the entire validation set (best epoch)
    np.concatenate(pred_seq),    # val_pred is the prediction for the entire validation set (best epoch)
    num_classes = 5
    )
        
    with tf.Session():
        print('Prediction 1 (amino acid locations) confusion matrix:')
        print(tf.Tensor.eval(conf_mat, feed_dict=None, session=None), '\n')
        
    ### Confusion matrices for prediction 2 (protein type)
    # The actual class is in the rows and predicted class is in the columns.
    # We utilize data from the best epoch (lowest validation loss)
    
    conf_mat = tf.confusion_matrix(
    true_type,    # val_true2 is the true y for the entire validation set (best epoch)
    pred_type,    # val_pred2 is the prediction for the entire validation set (best epoch)
    num_classes = 4
    )
    
    with tf.Session():
        print('Prediction 2 (type) confusion matrix:')
        print(tf.Tensor.eval(conf_mat, feed_dict=None, session=None), '\n')
