import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    wrong_predicted = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        wrong_predicted += 1
        dW[:, j] += X[i, :]
    dW[:, y[i]] -= wrong_predicted * X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW = dW / num_train + 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  delta = 1.0

  #score_kj = sum_n [x_kn * w_nj]
  scores = X.dot(W)
  scores_correct_class = scores[np.arange(num_train), y]
  #margins_kj = max(0, s_kj - s_k_(y_k) + delta)
  margins = np.maximum(0, scores - scores_correct_class[:, np.newaxis] + delta)
  loss = (margins.sum() - delta * num_train) / num_train + reg * np.sum(W*W)

  margins_binary = np.vectorize(lambda x: 1. if x>0 else 0)(margins)
  margins_binary[np.arange(num_train), y] = 0
  margins_binary[np.arange(num_train), y] = -margins_binary.sum(axis=1)

  dW = X.T.dot(margins_binary) / num_train + 2 * reg * W
  
  return loss, dW
