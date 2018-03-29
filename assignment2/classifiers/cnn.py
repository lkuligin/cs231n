from builtins import object
import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
         hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
         dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim

    self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, C, filter_size, filter_size])
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(0, weight_scale, [np.int(H/2)**2*num_filters, hidden_dim])
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    layer1_output, layer1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    layer2_activation_input, layer2_activation_cache = affine_forward(layer1_output, W2, b2)
    layer2_output, layer2_relu_cache = relu_forward(layer2_activation_input)

    layer3_out, layer3_cache = affine_forward(layer2_output, W3, b3)

    if y is None:
      return layer3_out

    grads = {}
    loss, grad_loss = softmax_loss(layer3_out, y)

    grad_x3, grad_w3, grad_b3 = affine_backward(grad_loss, layer3_cache)
    grads['W3'], grads['b3'] = grad_w3 + self.reg * W3, grad_b3
    
    grad_layer2_relu = relu_backward(grad_x3, layer2_relu_cache)
    grad_x2, grad_w2, grad_b2 = affine_backward(grad_layer2_relu, layer2_activation_cache)
    grads['W2'], grads['b2'] = grad_w2 + self.reg * W2, grad_b2

    _, dw1, db1 = conv_relu_pool_backward(grad_x2, layer1_cache)
    grads['W1'], grads['b1'] = dw1 + self.reg * W1, db1
    
    loss += self.reg * 0.5 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

    return loss, grads
