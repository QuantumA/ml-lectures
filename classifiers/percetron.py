import numpy as np


class Perceptron:
    weights = None
    bias = None
    errors = []

    def __init__(self, eta=0.01, random_state=1):
        self.eta = eta
        self.random_state = random_state
        self.errors = []

    def fit(self, features, labels, epochs):

        initializer = np.random.RandomState(self.random_state)
        self.weights = initializer.normal(loc=0.0, scale=0.01, size=features.shape[1])
        self.bias = np.float_(0.)

        for epoch in range(epochs):
            errors = 0
            for x_i, y_i in zip(features, labels):
                update = (y_i - self.predict(features=x_i)) * self.eta
                self.weights += update * x_i
                self.bias += update

                errors += int(update != 0.0)
            self.errors.append(errors)

        return self

    def _net_input(self, X):
        return X @ self.weights + self.bias

    def predict(self, features):
        return np.where(self._net_input(X=features) >= 0.0, 1, 0)


if __name__ == '__main__':
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=100, random_state=123)

    ppn = Perceptron(eta=0.05, random_state=1)
    ppn.fit(X, y, epochs=50)

