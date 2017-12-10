import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib import rnn
from tensorflow.python.ops import tensor_array_ops as ta_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn as rnn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

class CrfForwardRnnCell(core_rnn_cell.RNNCell):
    """Computes the alpha values in a linear-chain CRF.
    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
    """

    def __init__(self, transition_params):
        """Initialize the CrfForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary potentials.
              This matrix is expanded into a [1, num_tags, num_tags] in preparation
              for the broadcast summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous alpha
              values.
          scope: Unused variable scope of this cell.
        Returns:
          new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
              values containing the new alpha values.
        """
        state = array_ops.expand_dims(state, 2)

        transition_scores = state + self._transition_params
        new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])

        return new_alphas, new_alphas




class CrfBackwardRnnCell(core_rnn_cell.RNNCell):
    """Computes the alpha values in a linear-chain CRF.
    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
    """

    def __init__(self, transition_params):
        """Initialize the CrfForwardRnnCell.
        Args:
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfForwardRnnCell.
        Args:
        inputs: A [batch_size, num_tags] matrix of unary potentials.
        state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
        scope: Unused variable scope of this cell.
        Returns:
        new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
          values containing the new alpha values.
        """
        
        transition_scores = tf.expand_dims(inputs + state, 1) + self._transition_params
        new_betas = math_ops.reduce_logsumexp(transition_scores, [2])
        
        
        return new_betas, new_betas

def log_marginal(inputs, sequence_lengths, transition_params):
    """Computes the normalization for a CRF.
    Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
    to use as input to the CRF layer.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
    log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    # Split up the first and rest of the inputs in preparation for the forward
    # algorithm.
    first_input1 = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = array_ops.squeeze(first_input1, [1])
    rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])

    forward_cell = CrfForwardRnnCell(transition_params)
    backward_cell = CrfBackwardRnnCell(transition_params)

    (all_alphas, all_betas), (alphas, betas) = rnn_ops.bidirectional_dynamic_rnn(
        cell_fw=forward_cell,cell_bw=backward_cell, inputs=rest_of_input,
        sequence_length=sequence_lengths - 1, initial_state_fw=first_input,
        dtype=dtypes.float64)
    
    all_alphas = tf.concat([first_input1, all_alphas],1)
    all_betas = tf.concat([all_betas,tf.expand_dims(tf.zeros(tf.shape(alphas), dtype=tf.float64),1)],1)
    log_norm = tf.expand_dims(tf.expand_dims(math_ops.reduce_logsumexp(alphas, [1]), 1),2)
    log_marg = all_alphas + all_betas - log_norm

    return log_marg