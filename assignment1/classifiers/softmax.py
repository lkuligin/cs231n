import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  for i in xrange(num_train):
    log_denominator = 0.0
    scores = X[i].dot(W)
    scores -= np.max(scores)

    correct_class_score = scores[y[i]]

    log_denominator = np.sum(np.exp(scores))
    loss += np.log(log_denominator)
  
    for j in xrange(num_classes):
      if j == y[i]:
        loss -= scores[j]
        dW[:, j] += - X[i, :]
      dW[:, j] += (np.exp(scores[j]) / log_denominator) * X[i, :]

  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  scores = X.dot(W)
  #scaling to get numerical stability
  scores = np.apply_along_axis(lambda x: x-np.max(x), 1, scores)
  scores_correct_class = scores[np.arange(num_train), y]

  log_denominator = np.apply_along_axis(lambda x: np.sum(np.exp(x)), 1, scores)
  loss = -np.sum(scores_correct_class) + np.sum(np.log(log_denominator))
  loss = loss / num_train + reg * np.sum(W * W)

  dW = X.T.dot(np.exp(scores) / log_denominator[:, np.newaxis])

  correct_classes = np.zeros_like(scores)
  correct_classes[np.arange(num_train), y] = -1.

  dW += X.T.dot(correct_classes)

  dW = dW/ num_train + 2 * reg * W

  return loss, dW

