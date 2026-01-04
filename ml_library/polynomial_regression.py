import numpy as np
import matplotlib.pyplot as plt


class PolynomialRegression:
    def __init__(self, degree=2, lr=0.01, epochs=2000, normalize=False):
        self.degree = degree
        self.lr = lr
        self.epochs = epochs
        self.normalize = normalize
        self.theta = None
        self.cost_history = []

    def _poly_features(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        features = [np.ones((X.shape[0], 1))]

        for d in range(1, self.degree + 1):
            features.append(X ** d)

        return np.hstack(features)

    def fit(self, X, y, verbose=False):
        X_poly = self._poly_features(X)
        y = np.asarray(y).reshape(-1, 1)

        m, n = X_poly.shape
        self.theta = np.zeros((n, 1))

        for i in range(self.epochs):
            y_pred = X_poly @ self.theta
            error = y_pred - y

            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)

            grad = (1 / m) * (X_poly.T @ error)
            self.theta -= self.lr * grad

            if verbose and i % 200 == 0:
                print(f"Epoch {i}, Cost = {cost:.4f}")

        return self

    def predict(self, X):
        X_poly = self._poly_features(X)
        return X_poly @ self.theta

    def score(self, X, y):
        y = np.asarray(y).flatten()
        y_pred = self.predict(X).flatten()

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot

    def plot_cost_curve(self, Title="Cost Curve"):
        plt.figure()
        plt.plot(self.cost_history)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title(Title)
        plt.show()

    def plot_data_points(self, X, y, y_pred=None, Title="Predictions vs Actual"):
        plt.figure()
        plt.scatter(X, y, label="Actual")
        if y_pred is not None:
            plt.scatter(X, y_pred, label="Predicted")
        plt.legend()
        plt.title(Title)
        plt.show()
