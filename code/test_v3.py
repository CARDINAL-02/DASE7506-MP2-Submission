"""Tests guard safe arithmetic evaluation and portable tree traversal, not benchmark scores."""
import unittest
from fractions import Fraction
import numpy as np
from v3_features import safe_number, arithmetic_stats
from v3_method import tree_predict


class ArithmeticTests(unittest.TestCase):
    def test_safe_arithmetic_exact_and_rejects_code(self):
        self.assertEqual(safe_number('(12-2)/4'),Fraction(5,2))
        for text in ("__import__('os').getcwd()",'2**100000000','1/0','x+1'):
            self.assertIsNone(safe_number(text))

    def test_checks_correct_and_wrong_local_equations(self):
        a=arithmetic_stats('First 2 + 3 = 5. Then 8 / 2 = 5. #### 5')
        self.assertEqual(a['checked'],2)
        self.assertEqual(a['correct'],1)
        self.assertEqual(a['wrong'],1)
        self.assertEqual(arithmetic_stats('No equation #### 5')['checked'],0)

    def test_tree_numeric_threshold_and_leaf_value(self):
        tree={'feature':np.array([0,0,0]),'threshold':np.array([1.,0.,0.]),
              'left':np.array([1,0,0]),'right':np.array([2,0,0]),
              'leaf':np.array([False,True,True]),'value':np.array([0.,-2.,3.])}
        np.testing.assert_array_equal(tree_predict(np.array([[1.],[2.],[-3.]]),tree),[-2.,3.,-2.])


if __name__=='__main__':
    unittest.main()
