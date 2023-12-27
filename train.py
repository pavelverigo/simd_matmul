import numpy as np
from tqdm import trange
np.set_printoptions(suppress=True)
print(np.show_config())

X_train = np.fromfile("datasets/X_train.bin", dtype=np.uint8).reshape(-1, 28, 28)
Y_train = np.fromfile("datasets/Y_train.bin", dtype=np.uint8)
X_test = np.fromfile("datasets/X_test.bin", dtype=np.uint8).reshape(-1, 28, 28)
Y_test = np.fromfile("datasets/Y_test.bin", dtype=np.uint8)

def logsumexp(x):
    # return np.log(np.exp(x).sum(axis=1))

    # http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    c = x.max(axis=1)
    return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))

def forward_backward(x, y):
    # training
    out = np.zeros((len(y),10), np.float32)
    out[range(out.shape[0]),y] = 1

    # forward pass
    x_l1 = x.dot(l1)
    x_relu = np.maximum(x_l1, 0)
    x_l2 = x_relu.dot(l2)
    x_lsm = x_l2 - logsumexp(x_l2).reshape((-1, 1))
    x_loss = (-out * x_lsm).mean(axis=1)

    # training in numpy (super hard!)
    # backward pass

    # will involve x_lsm, x_l2, out, d_out and produce dx_sm
    d_out = -out / len(y)

    # derivative of logsoftmax
    # https://github.com/torch/nn/blob/master/lib/THNN/generic/LogSoftMax.c
    dx_lsm = d_out - np.exp(x_lsm)*d_out.sum(axis=1).reshape((-1, 1))

    # derivative of l2
    d_l2 = x_relu.T.dot(dx_lsm)
    dx_relu = dx_lsm.dot(l2.T)

    # derivative of relu
    dx_l1 = (x_relu > 0).astype(np.float32) * dx_relu

    # derivative of l1
    d_l1 = x.T.dot(dx_l1)
    
    return x_loss, x_l2, d_l1, d_l2

def layer_init(m, h):
  # gaussian is strong
  #ret = np.random.randn(m,h)/np.sqrt(m*h)
  # uniform is stronger
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

# reinit
np.random.seed(22)

INNER_LAYER_SZ = 128

l1 = layer_init(28*28, INNER_LAYER_SZ)
l2 = layer_init(INNER_LAYER_SZ, 10)

lr = 0.001
BS = 128
for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    X = X_train[samp].reshape((-1, 28*28))
    Y = Y_train[samp]
    x_loss, x_l2, d_l1, d_l2 = forward_backward(X, Y)

    # https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    l1 = l1 - lr*d_l1
    l2 = l2 - lr*d_l2

    cat = np.argmax(x_l2, axis=1)
    accuracy = (cat == Y).mean()
    loss = x_loss.mean()
    t.set_description(f"loss {loss:.5f} accuracy {accuracy:.5f}")

def forward(x):
    x = x.dot(l1)
    x = np.maximum(x, 0)
    x = x.dot(l2)  
    return x

def numpy_train_eval():
    Y_train_preds_out = forward(X_train.reshape((-1, 28*28)))
    Y_train_preds = np.argmax(Y_train_preds_out, axis=1)
    return (Y_train == Y_train_preds).mean()

def numpy_test_eval():
    Y_test_preds_out = forward(X_test.reshape((-1, 28*28)))
    Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
    return (Y_test == Y_test_preds).mean()

import time
start = time.time()
print(f"train accuracy {numpy_train_eval():.7f}")
print(f"elapsed {time.time() - start}")

print(f"test accuracy {numpy_test_eval():.7f}")

l1.tofile("weights/l1.bin")
l2.tofile("weights/l2.bin")
