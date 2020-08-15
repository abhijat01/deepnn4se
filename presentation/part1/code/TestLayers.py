import unittest
import MultLayer as m
import AddLayer as a


class LayerTest(unittest.TestCase):

    def test_multiplication(self):
        m_layer = m.MultiplicationLayer1D()
        _ = m_layer.forward(2, 4)
        x_grad, w_grad = m_layer.backprop(1.0)
        print("x_grad, w_grad = ({},{})".format(x_grad, w_grad))

    def test_addition(self):
        m_layer = a.AdditionLayer()
        _ = m_layer.forward(2, 4)
        x_grad, w_grad = m_layer.backprop(1.0)
        print("x_grad, w_grad = ({},{})".format(x_grad, w_grad))

if __name__ == '__main__':
    unittest.main()
