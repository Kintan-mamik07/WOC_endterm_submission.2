import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.01, seed=42):
        
        np.random.seed(seed)
        self.lr = lr
        self.params = {}
        self.L = len(layer_sizes) - 1   # number of layers (excluding input)

        # Initialize weights and biases
        for i in range(1, len(layer_sizes)):
            self.params[f"W{i}"] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01   #scaling down the initial weights
            self.params[f"b{i}"] = np.zeros((1, layer_sizes[i]))

    # -------- helper activations --------
    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)  # numerical stability
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    # -------- forward pass --------
    def forward(self, X):
         cache = {"A0": X}

    # hidden layers (ReLU)
         for i in range(1, self.L):
             Z = cache[f"A{i-1}"] @ self.params[f"W{i}"] + self.params[f"b{i}"]
             A = self.relu(Z)
             cache[f"Z{i}"], cache[f"A{i}"] = Z, A

        # Output layer
         ZL = cache[f"A{self.L-1}"] @ self.params[f"W{self.L}"] + self.params[f"b{self.L}"]
         AL = self.softmax(ZL)

         cache[f"Z{self.L}"], cache[f"A{self.L}"] = ZL, AL
         return AL, cache

    # -------- loss --------
    def compute_loss(self, Y_true_onehot, Y_pred):
        m = Y_true_onehot.shape[0]
        eps = 1e-9
        return -np.mean(np.sum(Y_true_onehot * np.log(Y_pred + eps), axis=1))

    # -------- backward pass --------
    def backward(self, cache, Y_true_onehot):
        grads = {}
        m = Y_true_onehot.shape[0]

        AL = cache[f"A{self.L}"]
        dZL = AL - Y_true_onehot

        grads[f"dW{self.L}"] = (cache[f"A{self.L-1}"].T @ dZL) / m
        grads[f"db{self.L}"] = np.mean(dZL, axis=0, keepdims=True)

        dA_prev = dZL

        for i in reversed(range(1, self.L)):
            dZ = dA_prev @ self.params[f"W{i+1}"].T * self.relu_deriv(cache[f"Z{i}"])
            grads[f"dW{i}"] = (cache[f"A{i-1}"].T @ dZ) / m
            grads[f"db{i}"] = np.mean(dZ, axis=0, keepdims=True)
            dA_prev = dZ

        return grads

    # -------- update --------
    def update(self, grads):
        for i in range(1, self.L + 1):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]

    # -------- training --------
    def fit(self, X, y, epochs=200, batch_size=32, verbose=True):
        m = X.shape[0]
        n_classes = int(y.max()) + 1
        Y_onehot = np.eye(n_classes)[y]
        Y_onehot=Y_onehot.squeeze()
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            Y_shuffled = Y_onehot[indices]

            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                Y_pred, cache = self.forward(X_batch)
                grads = self.backward(cache, Y_batch)
                self.update(grads)

            if verbose and (epoch + 1) % 20 == 0:
                Y_pred_full, _ = self.forward(X)
                loss = self.compute_loss(Y_onehot, Y_pred_full)
                print(f"Epoch {epoch+1}/{epochs}  |  Loss: {loss:.4f}")

    # -------- prediction --------
    def predict(self, X):
        Y_pred, _ = self.forward(X)
        return np.argmax(Y_pred, axis=1)
