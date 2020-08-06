import unittest


class MyDUMPTestCase(unittest.TestCase):
    """
    This is a really dummies test case which help to check if all import package work or not
    """
    def test_just_true(self):

        import csef
        import csef.utils
        import csef.utils.ensembles
        import csef.utils.visualization

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
