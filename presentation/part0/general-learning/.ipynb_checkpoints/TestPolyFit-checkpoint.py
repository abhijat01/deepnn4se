import unittest
import numpy as np
import torch.nn as nn
import torch as torch
import torch.optim as optim
import poly_fit as pf



class MyTestCase(unittest.TestCase):
    def test_polyfit(self):
        coeff = torch.tensor([1,3.,0,2], requires_grad=True)
        poly = pf.PolynomialFit(coeff.shape[0],init_coeff=coeff)
        value = poly.value(2.)

        print(value)

    def test_poly_layer_fwd(self):
        p_layer = pf.PolynomialLayer(3)
        value = p_layer.forward(np.array([2., 3]))
        print(value)
        x = np.array([1,2,3,4]).reshape((4,1))
        y = linear(2,1, x)
        print(x)
        print(y)

    def test_train_network(self):
        poly = pf.PolynomialLayer(2)
        x = np.array([1.,2.,3,4])
        y_target = torch.from_numpy( linear(2,1, x))
        net = poly
        optimizer = optim.SGD(net.parameters(), lr=.01)
        loss_func = nn.MSELoss()
        for i in range(1000):
            optimizer.zero_grad()
            y_pred  = net(x)
            loss = loss_func(y_pred, y_target)
            loss.backward()
            optimizer.step()
            if i%100 == 0 :
                print(loss)
                net.print_weights()



def linear(a,b, x):
    return a*x+b


if __name__ == '__main__':
    unittest.main()
