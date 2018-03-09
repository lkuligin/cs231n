import os
import sys
import unittest

import numpy as np

sys.path.insert(0,'..')
sys.path.insert(0,'../../../utils')
from neural_net import TwoLayerNet
import test_utils


class NeuralNetTest(test_utils.TestCaseWithParams):
  def setUp(self):
    self.net = self.kwargs['net']
    self.x = self.kwargs['x']
    self.y = self.kwargs['y']
    self.expected_loss_nt = self.kwargs['expected_loss_nt']
    self.expected_loss_wt = self.kwargs['expected_loss_wt']

  def test_neural_net_loss_notarget(self):
    scores = self.net.loss(self.x)
    np.testing.assert_allclose(scores, self.expected_loss_nt, 1e-04)

  def test_neural_net_loss_withtarget(self):
    loss, _ = self.net.loss(self.x, self.y, reg=0.05)
    np.testing.assert_allclose(loss, self.expected_loss_wt, 1e-04)

  def test_neural_net_gradient(self):
    loss, grads = self.net.loss(self.x, self.y, reg=0.0)
    f = lambda w: self.net.loss(self.x, self.y, reg=0.0)[0]
    for param_name, grad in grads.iteritems():
      grad_numerical = test_utils.eval_numerical_gradient(f, self.net.params[param_name], verbose=False)
      np.testing.assert_allclose(grad_numerical, grad, 1e-04)
    

if __name__ == '__main__':
  suite = unittest.TestSuite()
  np.random.seed(0)
  test_net = TwoLayerNet(4, 10, 3, std=1e-1)
  np.random.seed(1)
  x = 10 * np.random.randn(5, 4)
  y = np.array([0, 1, 2, 2, 1])
  expected_loss = np.asarray([[-0.81233741, -1.27654624, -0.70335995],
    [-0.17129677, -1.18803311, -0.47310444],
    [-0.51590475, -1.01354314, -0.8504215 ],
    [-0.15419291, -0.48629638, -0.52901952],
    [-0.00618733, -0.12435261, -0.15226949]])

  suite.addTest(test_utils.TestCaseWithParams.get_suite(NeuralNetTest, kwargs={'net': test_net, 'x': x, 'y': y, 
    'expected_loss_nt': expected_loss, 'expected_loss_wt': 1.303788}))
  unittest.TextTestRunner(verbosity=2).run(suite)
