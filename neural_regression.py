import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)


class NeuralNetwork:
    """
    Generic N-layer neural network for regression
    layer_sizes example: [n_features, 64, 32, 1]
    """
    def __init__(self, layer_sizes, lr=0.001, seed=42):
        np.random.seed(seed)

        self.L = len(layer_sizes) - 1
        self.lr = lr

        self.W = []
        self.b = []

        # He initialization for hidden layers
        for i in range(self.L):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.W.append(w)
            self.b.append(b)

        self.z = []
        self.a = []

    def forward(self, X):
        self.z = []
        self.a = [X]

        for i in range(self.L):
            z = self.a[-1] @ self.W[i] + self.b[i]

            # Last layer = linear (regression output)
            if i == self.L - 1:
                a = z
            else:
                a = relu(z)

            self.z.append(z)
            self.a.append(a)

        return self.a[-1]

    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

    def backward(self, y_true):
        m = len(y_true)

        dZ = self.a[-1] - y_true  # derivative of MSE wrt output

        grads_W = [None] * self.L
        grads_b = [None] * self.L

        for i in reversed(range(self.L)):
            grads_W[i] = (self.a[i].T @ dZ) / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA = dZ @ self.W[i].T
                dZ = dA * relu_prime(self.z[i-1])

        for i in range(self.L):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    def fit(self, X, y, epochs=200, verbose=True):
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        for e in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(y)

            if verbose and (e % max(1, epochs // 10) == 0):
                print(f"Epoch {e:04d} | Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)
