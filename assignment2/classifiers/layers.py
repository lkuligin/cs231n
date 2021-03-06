from builtins import range
import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx = dout.dot(w.T).reshape(x.shape)
  dw = np.dot(x.reshape(x.shape[0], w.shape[0]).T, dout)
  db = np.sum(dout, axis = 0)
  
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  return np.maximum(0, x), x


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  
  return np.where(cache > 0, dout, 0)


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the
  mean and variance of each feature, and these averages are used to normalize
  data at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7
  implementation of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  update_running_stat = lambda running, sample: momentum * running + (1-momentum) * sample
  scale = lambda val, avg, var: (val-avg) / np.sqrt(var + eps)

  out, cache = None, None
  if mode == 'train':
    x_mean = np.mean(x, axis=0)
    x_var = np.var(x, axis=0)
    x_scaled = scale(x, x_mean, x_var)
    out = gamma * x_scaled + beta

    runninig_mean = update_running_stat(running_mean, x_mean)
    running_var = update_running_stat(running_var, x_var)

    cache = {'x_scaled_mean': x-x_mean, 'x_scaled': x_scaled, 'gamma': gamma}
    cache['std'] =  np.sqrt(x_var + eps)
    cache['1_std'] = 1./np.sqrt(x_var + eps)
  elif mode == 'test':
    x_scaled = scale(x, running_mean, running_var)
    out = gamma * x_scaled + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.

  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.

  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.

  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None

  N, D = dout.shape
  x_scaled_mean = cache['x_scaled_mean']

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * cache['x_scaled'], axis=0)

  # http://cthorey.github.io./backpropagation/
  dx =(1. / N) * cache['gamma'] * 1 / cache['std'] * ((N * dout) - 
    np.sum(dout, axis=0) - (x_scaled_mean) * np.square(cache['1_std']) * np.sum(dout * (x_scaled_mean), axis=0))

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
    if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
    function deterministic, which is needed for gradient checking but not
    in real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask

  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']

  dx = None
  if mode == 'train':
    dx = dout * mask
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and
  width W. We convolve each input with F different filters, where each filter
  spans all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
    horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """  
  stride = conv_param['stride']
  pad = conv_param['pad']

  N, C, H, W = x.shape
  F, _, HH, WW = w.shape

  out_H = 1 + (H + 2 * pad - HH) / stride
  out_W = 1 + (H + 2 * pad - WW) / stride
  out = np.zeros((N, F, out_H, out_W))
  x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')

  for image_ind in np.arange(N):
    for filter_ in np.arange(F):
      for pos_h in np.arange(out_H):
        for pos_w in np.arange(out_W):
          pos_x_h = pos_h * stride
          pos_x_w = pos_w * stride
          conv_piece = x_padded[image_ind, :, pos_x_h : pos_x_h+HH, pos_x_w : pos_x_w+WW]
          out[image_ind, filter_, pos_h, pos_w] = np.sum(conv_piece * w[filter_, :]) + b[filter_]

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  _, _, HH_out, WW_out = dout.shape

  x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')

  dx_padded, dw, db = np.zeros_like(x_padded), np.zeros_like(w), np.zeros_like(b)
  
  for filter_ in np.arange(F):
    for channel in np.arange(C):
      for pos_h in np.arange(HH):
        for pos_w in np.arange(WW):
          conv_piece = x_padded[:, channel, pos_h : pos_h+HH_out*stride : stride, pos_w : pos_w+WW_out*stride : stride]
          dw[filter_, channel, pos_h, pos_w] = np.sum(dout[:, filter_, :, :] * conv_piece)

  for filter_ in np.arange(F):
    db[filter_] = np.sum(dout[:, filter_, :, :])

  for image_ind in np.arange(N):
    for filter_ in np.arange(F):
      for pos_h in np.arange(HH_out):
        for pos_w in np.arange(WW_out):
          dx_padded[image_ind, :, pos_h*stride : pos_h*stride+HH, pos_w*stride : pos_w*stride+WW] += dout[image_ind, filter_, pos_h, pos_w] * w[filter_, :]

  return dx_padded[:, :, pad:pad+H, pad:pad+W], dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = 1 + (H - pool_height) / stride
  W_out = 1 + (H - pool_width) / stride

  out = np.zeros((N, C, H_out, W_out))

  for image_ind in np.arange(N):
    for channel in np.arange(C):
      for h_pos in np.arange(H_out):
        for v_pos in np.arange(W_out):
          out[image_ind, channel, h_pos, v_pos] = np.max(x[image_ind, channel, h_pos*stride : h_pos*stride+pool_height, v_pos*stride : v_pos*stride+pool_width])

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  N, C, H, W = x.shape
  _, _, HH, WW = dout.shape
  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']

  dx = np.zeros_like(x)

  for image_ind in np.arange(N):
    for channel in np.arange(C):
      for pos_h in np.arange(HH):
        for pos_v in np.arange(WW):
          max_index = np.argmax(x[image_ind, channel, pos_h*stride : pos_h*stride+pool_height, pos_v*stride : pos_v*stride+pool_width])
          index = np.unravel_index(max_index, [pool_height,pool_width])
          dx[image_ind, channel, pos_h*stride : pos_h*stride+pool_height, pos_v*stride : pos_v*stride+pool_width][index] = dout[image_ind, channel, pos_h, pos_v]

  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
    old information is discarded completely at every time step, while
    momentum=1 means that new information is never incorporated. The
    default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  N, C, H, W = x.shape

  x_flattened = x.transpose((0,2,3,1)).reshape(-1, C)
  out_flattened, cache = batchnorm_forward(x_flattened, gamma, beta, bn_param)


  return out_flattened.reshape((N, H, W, C)).transpose((0,3,1,2)), cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  N, C, H, W = dout.shape
  dout_flattened = dout.transpose((0,2,3,1)).reshape(-1, C)
  
  dx_flattened, dgamma, dbeta = batchnorm_backward(dout_flattened, cache)
  
  return dx_flattened.reshape((N, H, W, C)).transpose((0,3,1,2)), dgamma, dbeta


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  shifted_logits = x - np.max(x, axis=1, keepdims=True)
  Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
  log_probs = shifted_logits - np.log(Z)
  probs = np.exp(log_probs)
  N = x.shape[0]
  loss = -np.sum(log_probs[np.arange(N), y]) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
