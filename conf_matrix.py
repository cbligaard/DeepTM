##### Confusion matrices

# Import modules
import tensorflow as tf
import numpy as np

def conf_matrix(true_seq, pred_seq, true_type, pred_type):
    
    ### Confusion matrix for prediction 1 (position of each amino acid)
    # The actual class is in the rows and predicted class is in the columns.
    # We utilize data from the best epoch (lowest validation loss)

    conf_mat = tf.confusion_matrix(
    np.concatenate(true_seq),    # val_true is the true y for the entire validation set (best epoch)
    np.concatenate(pred_seq),    # val_pred is the prediction for the entire validation set (best epoch)
    num_classes = 5
    )

    with tf.Session():
        conf1 = tf.Tensor.eval(conf_mat, feed_dict=None, session=None)
    
    print('Confusion matrix for classification of amino acid location')
    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format('', 'In->Out', 'Out->In', 'SP', 'Inside', 'Outside'))
    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format('In->Out', *conf1[0]))
    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format('Out->In', *conf1[1]))
    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format('SP', *conf1[2]))
    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format('Inside', *conf1[3]))
    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n'.format('Outside', *conf1[4]))

    # In percentage

    true_count1 = np.bincount(np.concatenate([batch.flatten() for batch in true_seq]).ravel().tolist())

    percentConf1 = []
    for i in range(len(conf1)):
        percent = []
        for j in range(len(conf1[i])):
            percent.append(conf1[i][j]/true_count1[i]*100)
        percentConf1.append(percent)

    print('Confusion matrix for classification of amino acid location in percent')
    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format('', 'In->Out', 'Out->In', 'SP', 'Inside', 'Outside'))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}'.format('In->Out', *percentConf1[0]))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}'.format('Out->In', *percentConf1[1]))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}'.format('SP', *percentConf1[2]))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}'.format('Inside', *percentConf1[3]))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}\n'.format('Outside', *percentConf1[4]))
    
    ### Confusion matrices for prediction 2 (protein type)
    # The actual class is in the rows and predicted class is in the columns.
    # We utilize data from the best epoch (lowest validation loss)
    
    conf_mat = tf.confusion_matrix(
    true_type,    # val_true2 is the true y for the entire validation set (best epoch)
    pred_type,    # val_pred2 is the prediction for the entire validation set (best epoch)
    num_classes = 4
    )
    
    with tf.Session():
        conf2 = tf.Tensor.eval(conf_mat, feed_dict=None, session=None)
    
    print('Confusion matrix for classification of proteins')
    print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('', 'TM', 'SP+TM', 'SP+Glob', 'Glob'))
    print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('TM', conf2[2][2], conf2[2][1], conf2[2][0], conf2[2][3]))
    print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('SP+TM', conf2[1][2], conf2[1][1], conf2[1][0], conf2[1][3]))
    print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('SP+Glob', conf2[0][2], conf2[0][1], conf2[0][0], conf2[0][3]))
    print('{:<10} {:<10} {:<10} {:<10} {:<10}\n'.format('Glob', conf2[3][2], conf2[3][1], conf2[3][0], conf2[3][3]))
    
    
    # In percentage
    true_count = np.bincount(true_type)
            
    percentConf = []
    for i in range(len(conf2)):
        percent = []
        for j in range(len(conf2[i])):
            percent.append(conf2[i][j]/true_count[i]*100)
        percentConf.append(percent)
    
    print('Confusion matrix for classification of proteins in percent')
    print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('', 'TM', 'SP+TM', 'SP+Glob', 'Glob'))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}'.format('TM', percentConf[2][2], percentConf[2][1], percentConf[2][0], percentConf[2][3]))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}'.format('SP+TM', percentConf[1][2], percentConf[1][1], percentConf[1][0], percentConf[1][3]))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}'.format('SP+Glob', percentConf[0][2], percentConf[0][1], percentConf[0][0], percentConf[0][3]))
    print('{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}\n'.format('Glob', percentConf[3][2], percentConf[3][1], percentConf[3][0], percentConf[3][3]))