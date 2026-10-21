"""Information-boundary tests catch accidental use of reasoning in restricted ablations."""
import unittest
import numpy as np
from supplement_features import restricted


def bank(texts):
    return [{'id':'synthetic','question':'There are 2 boxes with 3 apples each. How many?',
             'candidates':[{'index':i,'text':t} for i,t in enumerate(texts)]}]


class BoundaryTests(unittest.TestCase):
    def test_answer_only_ignores_trace_style_and_reasoning(self):
        a = bank(['Short #### 6','#### 9','#### 6','#### 3'])
        b = bank(['However check '+('x = 99\n'*20)+'#### 6','#### 9','#### 6','#### 3'])
        np.testing.assert_array_equal(restricted(a,'answer'),restricted(b,'answer'))

    def test_surface_ignores_answer_values_with_same_format(self):
        a = bank(['Therefore #### 6','#### 9','#### 6','#### 3'])
        b = bank(['Therefore #### 8','#### 7','#### 5','#### 1'])
        np.testing.assert_array_equal(restricted(a,'surface'),restricted(b,'surface'))

    def test_no_embedding_input_required(self):
        x = restricted(bank(['#### 6']*4),'answer_surface')
        self.assertEqual(x.shape[0],4)
        self.assertTrue(np.isfinite(x).all())

    def test_listwise_gradient_and_invalid_mask(self):
        from supplement import objective
        x=np.array([[[1.,2.],[3.,1.],[9.,9.],[-1.,3.]]])
        target=np.array([[.5,.5,0.,0.]])
        valid=np.array([[True,True,False,True]])
        w=np.array([.2,-.3])
        loss,grad=objective(w,x,target,valid,.01)
        numerical=[]
        for j in range(2):
            step=np.zeros(2)
            step[j]=1e-6
            numerical.append((objective(w+step,x,target,valid,.01)[0]-objective(w-step,x,target,valid,.01)[0])/2e-6)
        np.testing.assert_allclose(grad,numerical,rtol=1e-6,atol=1e-7)
        x[0,2]=[1e6,-1e6]
        self.assertAlmostEqual(objective(w,x,target,valid,.01)[0],loss)


if __name__ == '__main__':
    unittest.main()
