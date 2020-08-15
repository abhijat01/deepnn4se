class MultiplicationLayer1D:
    def __init__(self):
        self.cache = {}

    def forward(self, x, w):
        self.cache['x'] = x
        self.cache['w'] = w
        return w * x

    def backprop(self, incoming_grad):
        x = self.cache['x']
        w = self.cache['w']
        return (w * incoming_grad,
                x * incoming_grad)
