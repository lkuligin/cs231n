import os
import sys
import unittest

import numpy as np

sys.path.insert(0,'..')
sys.path.insert(0,'../../../utils')
from linear_svm import svm_loss_naive, svm_loss_vectorized
import test_utils


class LinearSvmTest(test_utils.TestCaseWithParams):
  def setUp(self):
    self.weights = self.kwargs['W']
    self.x = self.kwargs['x']
    self.y = self.kwargs['y']
    self.reg = self.kwargs['reg']
    self.expected = self.kwargs['expected']

  def test_svm_loss_naive_loss(self):
    loss, _ = svm_loss_naive(self.weights, self.x, self.y, self.reg)
    np.testing.assert_allclose(loss, self.expected)

  def test_svm_loss_vectorized_loss(self):
    loss, _ = svm_loss_vectorized(self.weights, self.x, self.y, self.reg)
    np.testing.assert_allclose(loss, self.expected)

  def test_svm_loss_naive_gradient(self):
    loss, grad = svm_loss_naive(self.weights, self.x, self.y, self.reg)
    f = lambda w: svm_loss_naive(self.weights, self.x, self.y, self.reg)[0]
    grad_numerical = test_utils.grad_check_sparse(f, self.weights, grad)

  def test_svm_loss_vectorized(self):
    loss_naive, grad_naive = svm_loss_naive(self.weights, self.x, self.y, self.reg)
    loss_vect, grad_vect = svm_loss_vectorized(self.weights, self.x, self.y, self.reg)
    np.testing.assert_allclose(loss_naive, loss_vect)
    np.testing.assert_allclose(grad_naive, grad_vect)
    


if __name__ == '__main__':
  suite = unittest.TestSuite()
  w = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]])
  x = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6], [5,6,7,8]])
  y = np.array([0,2,1,2]).T
  suite.addTest(test_utils.TestCaseWithParams.get_suite(LinearSvmTest, kwargs={'W': w, 'x': x, 'y': y, 'reg': 0.0, 'expected': 1.95}))
  suite.addTest(test_utils.TestCaseWithParams.get_suite(LinearSvmTest, kwargs={'W': w, 'x': x, 'y': y, 'reg': 0.1, 'expected': 2.153}))
  unittest.TextTestRunner(verbosity=2).run(suite)
