class MultiplicationLayer1D:
    def __init__(self):
        self.cache = {}

    def forward(self, x, w):
        self.cache['x'] = x
        self.cache['w'] = w
        return w * x

    def backprop(self, incoming_grad):
        x_grad = self.cache['w']
        w_grad = self.cache['x']
        x_grad *= incoming_grad
        w_grad *= incoming_grad
        return x_grad, w_grad
