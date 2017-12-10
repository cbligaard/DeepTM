###### Building the Neural Network Model

### Loading libraries
import sys
import os
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn   
sys.path.append(os.path.join('.', '..')) 
import crf 
from marginal_probabilities import log_marginal



### Setting the neural network architecture

def model_setup(learning_rate = 0.0005, n_hidden = 20, CRF = False, act_fun = relu):
    # Resetting the graph
    tf.reset_default_graph()

    # Fixed settings
    n_classes = 5 # 0: Membrane in-out, 1: Membrane out-in, 2: signal peptide, 3: inside, 4: outside
    blosum = 23   # 23 amino acids encoded in BLOSUM
    n_types = 4   # 0: Signal peptide, 1: Signal peptide with transmembrane, 2: Transmembrane, 3: Globular

    # Defining placeholders
    X_pl = tf.placeholder(tf.float32, [None, None, blosum], name='X_input')
    mask_pl = tf.placeholder(tf.float32, [None, None], name='mask_input') 
    Y_pl = tf.placeholder(tf.int32, [None, None], name='Y_input') 
    Y_type_pl = tf.placeholder(tf.float32, [None, n_types], name='y_type')
    len_pl = tf.placeholder(tf.int32, [None], name='seq_length')
   
    ## MODEL FOR PREDICTION OF AMINO ACID LOCATION
    # Setting the forward cell to be an LSTM with n_hidden units
    f_cell = tf.nn.rnn_cell.LSTMCell(
        n_hidden, 
    )
    
    # Setting the backward cell to be an LSTM with n_hidden units
    b_cell = tf.nn.rnn_cell.LSTMCell(
        n_hidden, 
    )
    
    # Building the RNN   - maybe name it BiLSTM? 
    l_LSTM, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=f_cell, 
        cell_bw=b_cell,
        inputs=X_pl, 
        sequence_length=len_pl,
        dtype=tf.float32,
    )

    # Concatenating the output from the forward and backward run
    l_concat = tf.concat(
        values = l_LSTM,
        axis = -1,
        name = 'l_concat'
    )
    
    # Using a fully connected layer on top
    l_dense_seq = tf.layers.dense(
        inputs=l_concat,
        units = n_classes,
        activation = None,
        name = 'l_dense_seq'
    )
    
    # CRF 
    if CRF:
        with tf.variable_scope('CRF'): 
            # CRF function
            log_likelihood, transition_params = crf.crf_log_likelihood(
                inputs=l_dense_seq,
                tag_indices=Y_pl,
                sequence_lengths=len_pl,
                transition_params=None
            )
            
            # Working in CPU environment
            with tf.device('/cpu:0'):
                # Cast inputs to float64 (this is done due to some error in the rounding for the float32 - also this part of the program )
                l_dense_64 = tf.cast(l_dense_seq, dtype=tf.float64)
                len_pl_64 = tf.cast(len_pl, dtype=tf.int64)
                transition_params_64 = tf.cast(transition_params, dtype=tf.float64)

                # Calculating log-marginal probabilities 
                log_marginal_prob = log_marginal(
                inputs = l_dense_64,
                sequence_lengths = len_pl_64,
                transition_params = transition_params_64
                )
            
                # Cast output to float32 and take exponential function to get marginal probabilities (y_out_seq)
                log_marginal_prob = tf.cast(log_marginal_prob, dtype=tf.float32)
            
            # Take exponential function to get marginal probabilities (y_out_seq)
            y_out_seq = tf.exp(log_marginal_prob)
            
            # Decoding the highest scoring sequence of tags (only return if validating/testing)
            y_seq, viterbi_score = crf.crf_decode(
                potentials=l_dense_seq, 
                transition_params=transition_params, 
                sequence_length=len_pl
            )
            
            # Defining the loss
            loss_seq = tf.reduce_mean(-log_likelihood)
                
    # Without CRF
    else:
    
        # Using a softmax layer to get output in classes (used for calculating accuracy)
        y_out_seq = tf.nn.softmax(
            l_dense_seq, 
            name = 'y_out_seq'
        ) 
        
        # An extra argmax layer to get the actual predictions
        y_seq = tf.argmax(
            y_out_seq, 
            axis = -1, 
            name = 'y_seq'
        )
        
        # Defining the loss function for prediction of amino acid location
        loss_seq = tf.contrib.seq2seq.sequence_loss(
            y_out_seq, 
            Y_pl, 
            mask_pl)
        
        # Only used for CRF runs - too see how the matrix looks
        transition_params = tf.zeros([1], tf.int32)


        
## MODEL FOR PREDICTION OF PROTEIN TYPE
    # Making the y_out into a vector of size (batch_size, num_classes)
    newInput = tf.reduce_mean(
        y_out_seq,
        axis = 1, 
        name = 'newInput'
    ) 
    
    # Feed the newInput to a fully connected layer (non-linear)
    l_dense_type = tf.layers.dense(
        inputs = newInput,
        units = n_types,
        activation = exec(act_fun),
        name = 'l_dense_type'
    )
    
    # Making a softmax layer (for calculating accuracy)
    y_out_type = tf.nn.softmax(
        l_dense_type,
        name = 'y_out_type'
    )
    
    # An extra argmax layer to get the actual predictions
    y_type = tf.to_float(tf.argmax(
        y_out_type,
        axis = -1,
        name = 'y_type'
    ))
    

## DEFINING COST, TRAIN OP AND ACCCURACY
    # 1) Define cost function for prediction of type
    with tf.variable_scope('loss'):
        # Define loss function for y_type
        loss_type = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_type_pl, logits=l_dense_type))

        loss = loss_seq + loss_type 

    # 2) Define the training op
    with tf.variable_scope('trainOptimizer'):

        # Defining the optimizer
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)     # Gradient descent
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)                 # ADAM

        # Computes the gradients and applies them using minimize() function
        train_op = optimizer.minimize(loss)

    # 3) Define accuracy
    with tf.variable_scope('accuracy'):
        
        ## Calculating accuracy for predicting amino acid location in the sequence
        y_seq = tf.cast(y_seq, dtype=tf.int32)
        # Getting the confidence for the predicted class
        y_seq_conf = tf.to_float(tf.reduce_max(y_out_seq,-1))
        correct_seq = tf.to_float(tf.equal(y_seq, Y_pl)) * mask_pl
        accuracy_seq = tf.reduce_sum(correct_seq) / tf.reduce_sum(mask_pl)
        
        ## Calculating accuracy for prediction of protein type
        # Getting the confidence for the predicted class
        y_type_conf = tf.to_float(tf.reduce_max(y_out_type,-1))
        # Getting the true type as a single value 0,1,2 or 3 (to compare)
        y_true_type = tf.to_float(tf.argmax(Y_type_pl,-1))
        correct_type = tf.to_float(tf.equal(y_type, y_true_type))
        accuracy_type = tf.reduce_mean(correct_type)

    return(X_pl, mask_pl, Y_pl, len_pl, Y_type_pl, train_op, y_seq, y_seq_conf,  y_type, y_type_conf, loss, accuracy_seq, accuracy_type, transition_params)
