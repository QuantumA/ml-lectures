import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification

from visual.tools import plot_decision_regions


class LogisticRegressionGD:
    """LOgisticREgression form Adaline"""

    def __init__(self, lr=1e-2, random_state=1):
        self.lr = lr
        self.random_state = random_state
        self.w_initialized = False
        self.weights = None
        self.bias = None

    def fit(self, features, target, epochs):
        self._weight_initializer(features.shape[1])

        for epoch in range(epochs):
            net_input = self._net_input(X=features)
            output = self.activation(net_input)
            errors = (target - output)
            self.weights += self.lr * features.T @ errors / features.shape[0]
            self.bias += self.lr * errors.mean()
            # loss = −𝑦 log(𝜎(𝑧)) − (1−𝑦) log(1−𝜎(𝑧)
            loss = (- target.dot(np.log(output)) - ((1 - target).dot(np.log(1 - output))) / features.shape[0])
            self.losses.append(loss)

            print(f"Epoch {epoch} of {epochs} loss: {loss}")
        return self

    def _weight_initializer(self, shape: int):
        self.rnd_gen = np.random.RandomState(self.random_state)
        self.weights = self.rnd_gen.normal(loc=0.0, scale=0.01, size=shape)
        self.bias = np.float_(0.)
        self.losses = []

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rnd_gen.permutation(len(y))
        return X[r], y[r]

    def _net_input(self, X):
        return X @ self.weights + self.bias

    def predict(self, features):
        return np.where(self.activation(self._net_input(X=features)) >= 0.5, 1, 0)

    def activation(self, net_input):
        return 1. / (1. + np.exp(-np.clip(net_input, -250, 250)))

    @staticmethod
    def sigmoid(z):
        return 1. / (1. + np.exp(z))


if __name__ == "__main__":
    X, y = make_classification(n_features=2,
                               n_redundant=0,
                               n_informative=1,
                               n_clusters_per_class=1,
                               class_sep=1.2)

    alne_sgd = LogisticRegressionGD(lr=0.001, random_state=1)
    alne_sgd.fit(X, y, epochs=150000)

    print((y - alne_sgd.predict(X)))

    fig = plot_decision_regions(X=X, y=y, classifier=alne_sgd)
    plt.title("LogisticRegression")
    plt.show()

    plt.plot(range(1, len(alne_sgd.losses) + 1), alne_sgd.losses, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Average loss")
    plt.tight_layout()
    plt.show()
