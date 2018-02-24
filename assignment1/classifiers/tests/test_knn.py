import numpy as np
import os
import sys
import unittest

sys.path.insert(0,'..')
sys.path.insert(0,'../../../utils')
import knn
import test_utils


class KNearestNeighborTest(test_utils.TestCaseWithParams):

  def setUp(self):
    self.classifier = knn.KNearestNeighbor()
    self.classifier.train(np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]]), np.array([0,1,1]))

  def test_compute_distances_two_loops(self):
    dists = self.classifier.compute_distances_two_loops(self.kwargs['input'])
    np.testing.assert_allclose(dists, self.kwargs['dists'])

  def test_compute_distances_one_loop(self):
    dists = self.classifier.compute_distances_one_loop(self.kwargs['input'])
    np.testing.assert_allclose(dists, self.kwargs['dists'])

  def test_compute_distances_no_loops(self):
    dists = self.classifier.compute_distances_no_loops(self.kwargs['input'])
    np.testing.assert_allclose(dists, self.kwargs['dists'])

  def test_predict_labels(self):
    pred =  self.classifier.predict_labels(self.kwargs['dists'], 1)
    np.testing.assert_allclose(pred, self.kwargs['pred_k1'])
    pred =  self.classifier.predict_labels(self.kwargs['dists'], 2)
    np.testing.assert_allclose(pred, self.kwargs['pred_k2'])


if __name__ == '__main__':
  suite = unittest.TestSuite()
  test_case1 = {'dists': np.array([[2.,0,2], [6,4,2]]), 'input': np.array([[2,3,4,5], [4,5,6,7]]), 'pred_k1': np.array([1., 1]), 'pred_k2': np.array([0., 1])}
  suite.addTest(test_utils.TestCaseWithParams.get_suite(KNearestNeighborTest, kwargs=test_case1))
  unittest.TextTestRunner(verbosity=2).run(suite)
