from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  hidden_layer = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  activation = np.tanh(hidden_layer)
  cache = hidden_layer, x, prev_h, Wx, Wh
  return activation, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.

  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass

  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  hidden_layer, x, prev_h, Wx, Wh = cache
  
  dactivation = (1 - np.tanh(hidden_layer)**2) * dnext_h
  
  dx = np.dot(dactivation, Wx.T)
  dprev_h = np.dot(dactivation, Wh.T)
  dWx = np.dot(x.T, dactivation)
  dWh = np.dot(prev_h.T, dactivation)
  db = np.sum(dactivation, axis=0)
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.

  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  N, T, D = x.shape
  _, H = h0.shape

  layer_output = h0
  x_sequenced = x.transpose((1,0,2))
  h = np.zeros((T, N, H))
  
  cache = []

  for step in np.arange(T):
    sequence_element = x_sequenced[step, ...]
    layer_output, layer_cache = rnn_step_forward(sequence_element, layer_output, Wx, Wh, b)
    cache.append(layer_cache)
    h[step] = layer_output

  return h.transpose((1,0,2)), cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.

  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)

  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """


  N, T, H = dh.shape
  _, D = cache[0][1].shape

  dx = np.zeros((T, N, D))
  dprev_h = np.zeros((N, H))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros(H)

  dh_sequenced = dh.transpose((1, 0, 2))

  for step in np.arange(T)[::-1]:
    grad_h = dh_sequenced[step] + dprev_h
    dx[step], dprev_h, grad_Wx, grad_Wh, grad_b = rnn_step_backward(grad_h, cache[step])
    dWx += grad_Wx
    dWh += grad_Wh
    db += grad_b 

  return dx.transpose((1,0,2)), dprev_h, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.

  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.

  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out = W[x, :]
  return W[x, :], (x, W)


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.

  HINT: Look up the function np.add.at

  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass

  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  x, W = cache
  dW = np.zeros_like(W)
  np.add.at(dW, x, dout) 
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  _, H = prev_h.shape

  activation = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  input_gate = sigmoid(activation[:, :H])
  forget_gate = sigmoid(activation[:, H:2*H])
  output_gate = sigmoid(activation[:, 2*H:3*H])
  block_input = np.tanh(activation[:, 3*H:])

  next_c = forget_gate * prev_c + input_gate * block_input
  next_h = output_gate * np.tanh(next_c)

  cache = input_gate, forget_gate, output_gate, block_input, next_c, prev_c, prev_h, Wx, Wh, x

  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.

  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """

  input_gate, forget_gate, output_gate, block_input, next_c, prev_c, prev_h, Wx, Wh, x = cache

  dsigmoid = lambda x: x * (1-x)
  dtanh = lambda x: (1 - x ** 2)

  #dl/da = dl/dh * dh/da + dl/dc * dc/da = dl/dh * (do/da * np.tanh(next_c) + o * dt(c) * dc/da)+ dl/dc * (df/da * prev_c + di/da * block_input)

  dnext_h_dc = output_gate *  dtanh(np.tanh(next_c)) * dnext_h + dnext_c

  dinput_gate = dsigmoid(input_gate) * block_input * dnext_h_dc
  dforget_gate = dsigmoid(forget_gate) * prev_c * dnext_h_dc
  doutput_gate = dsigmoid(output_gate) * dnext_h * np.tanh(next_c)
  dblock_input = dtanh(block_input) * input_gate * dnext_h_dc

  dactivation = np.hstack((dinput_gate, dforget_gate, doutput_gate, dblock_input))
  dx = np.dot(dactivation, Wx.T)
  dprev_h = np.dot(dactivation, Wh.T)
  dWx = np.dot(x.T, dactivation)
  dWh = np.dot(prev_h.T, dactivation)
  db = np.sum(dactivation, axis=0)
  dprev_c = forget_gate * dnext_h_dc

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.

  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.

  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)

  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  N, T, D = x.shape
  _, H = h0.shape

  hidden_state = h0
  cell_state = np.zeros_like(hidden_state)

  x_sequenced = x.transpose((1,0,2))
  h = np.zeros((T, N, H))
  
  cache = []

  for step in np.arange(T):
    sequence_element = x_sequenced[step, ...]
    hidden_state, cell_state, step_cache = lstm_step_forward(sequence_element, hidden_state, cell_state, Wx, Wh, b)
    cache.append(step_cache)
    h[step] = hidden_state

  return h.transpose((1,0,2)), cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]

  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  N, T, H = dh.shape
  _, D = cache[0][-1].shape

  dx = np.zeros((T, N, D))
  dprev_h = np.zeros((N, H))
  grad_cell = np.zeros((N, H))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros(4*H)

  dh_sequenced = dh.transpose((1, 0, 2))

  for step in np.arange(T)[::-1]:
    grad_h = dh_sequenced[step] + dprev_h
    a, dprev_h, grad_cell, grad_Wx, grad_Wh, grad_b = lstm_step_backward(grad_h, grad_cell, cache[step])
    dx[step] = a
    dWx += grad_Wx
    dWh += grad_Wh
    db += grad_b 

  return dx.transpose((1, 0, 2)), dprev_h, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)

  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
     0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape

  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)

  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]

  if verbose: print('dx_flat: ', dx_flat.shape)

  dx = dx_flat.reshape(N, T, V)

  return loss, dx
