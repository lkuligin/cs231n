import unittest


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
