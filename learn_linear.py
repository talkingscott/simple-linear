"""
Learn a linear model.

estimator: y = w * x
loss: MSE 1/4 * sum(w**2*x[i]**2 - 2*w*x[i]*y[i] + y[i]**2)
loss gradient: 1/2 * (w*sum(x[i]**2) - sum(x[i]*y[i]))
optimizer: simple gradient descent (no momentum, etc.)
"""
import logging

X = [1., 2., 3., 4.]
Y = [0.9, 2.1, 3.2, 3.8]
W = 2.
LR = 0.01

def loss():
    total = 0.
    for x, y in zip(X, Y):
        total += (W * x - y)**2
    return total * .25

def loss_gradient():
    total1 = 0.
    total2 = 0.
    for x, y in zip(X, Y):
        total1 += W * (x**2)
        total2 += x * y
    return (total1 - total2) * .5

def train():
    global W
    # TODO: terminate based on loss
    for _ in range(50):
        current_loss = loss()
        current_gradient = loss_gradient()
        logging.debug('W: %.4f loss: %.4f gradient: %.4f', W, current_loss, current_gradient)
        W = W - (LR * current_gradient)

def _main():
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _main()
