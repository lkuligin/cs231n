from random import randrange
import unittest

import numpy as np
from past.builtins import xrange


class TestCaseWithParams(unittest.TestCase):
  """ TestCase classes that want to be parametrized should
    inherit from this class.
  """
  def __init__(self, methodName='runTest', kwargs={}):
    super(TestCaseWithParams, self).__init__(methodName)
    self.kwargs = kwargs

  @staticmethod
  def get_suite(testcase_class, kwargs={}):
    """ Create a suite containing all tests taken from the given
        subclass, passing them the parameter 'param'.
    """
    testloader = unittest.TestLoader()
    testnames = testloader.getTestCaseNames(testcase_class)
    suite = unittest.TestSuite()
    for name in testnames:
      suite.addTest(testcase_class(name, kwargs=kwargs))
    return suite


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5, printable=False):
  """
  sample a few random elements and only return numerical
  in this dimensions.
  """

  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in x.shape])

    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evaluate f(x + h)
    x[ix] = oldval - h # increment by h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    if printable:
      print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))
    else:
      np.testing.assert_allclose(grad_numerical, grad_analytic)
