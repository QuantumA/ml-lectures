import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification

from visual.tools import plot_decision_regions


class AdalineGD:
    """ADAptative LInear NEuron Gradient Descent"""
    weights = None
    bias = None
    losses = []

    def __init__(self, lr=1e-2, random_state=1):
        self.lr = lr
        self.random_state = random_state

    def fit(self, features, target, epochs):
        initializer = np.random.RandomState(self.random_state)
        self.weights = initializer.normal(loc=0.0, scale=0.01, size=features.shape[1])
        self.bias = np.float_(0.)

        for epoch in range(epochs):
            net_input = self.predict(features=features)
            output = self.activation(net_input)
            errors = (target - output)
            self.weights += self.lr * features.T @ errors
            self.bias += self.lr * errors.sum()
            loss = (errors ** 2).mean()
            self.losses.append(loss)

            print(f"Epoch {epoch} of {epochs} loss: {loss}")

        return self

    def _net_input(self, X):
        return X @ self.weights + self.bias

    def predict(self, features):
        return np.where(self._net_input(X=features) >= 0.0, 1, 0)

    def activation(self, net_input):
        return net_input


if __name__ == "__main__":
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)

    alne = AdalineGD(lr=0.05, random_state=1)
    alne.fit(X, y, epochs=500)

    fig = plot_decision_regions(X=X, y=y, classifier=alne)
    plt.title("Adaline")
