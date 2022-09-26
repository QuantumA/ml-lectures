import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification

from visual.tools import plot_decision_regions


class AdalineSGD:
    """ADAptative LInear NEuron with Stochastic Gradient Descent"""

    def __init__(self, lr=1e-2, random_state=1, shuffle=True, plot_evolution: bool = False):
        self.lr = lr
        self.random_state = random_state
        self.shuffle = shuffle
        self.w_initialized = False
        self.weights = None
        self.bias = None
        self.losses = []
        self.plot_evolution = plot_evolution

    def fit(self, features, target, epochs):
        self._weight_initializer(features.shape[1])
        for epoch in range(epochs):
            _losses = []
            if self.shuffle:
                features, target = self._shuffle(features, target)

            for xi, yi in zip(features, target):
                xi = xi.reshape(1, 2)
                _losses.append(self._update_weights(np.array(xi), yi))
            avg_loss = np.mean(_losses)
            self.losses.append(avg_loss)

            if self.plot_evolution:
                plt.figure(figsize=(16, 16))
                plt.subplot(4, 4, epoch + 1)
                plot_decision_regions(X=features, y=target, classifier=self)
                plt.grid(True)
                plt.title(f"Adaline SGD {epoch} loss {avg_loss}")

            print(f"Epoch {epoch} of {epochs} loss: {avg_loss}")
        plt.show()
        return self

    def _update_weights(self, features, target):
        """Apply Adaline learning rule to update the weights"""
        net_input = self.predict(features=features)
        output = self.activation(net_input)
        errors = (target - output)
        self.weights += self.lr * features.T @ errors
        self.bias += self.lr * errors.sum()
        loss = (errors ** 2).mean()
        return loss

    def _weight_initializer(self, shape: int):
        self.rnd_gen = np.random.RandomState(self.random_state)
        self.weights = self.rnd_gen.normal(loc=0.0, scale=0.01, size=shape)
        self.bias = np.float_(0.)
        self.w_initialized = True

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._weight_initializer(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rnd_gen.permutation(len(y))
        return X[r], y[r]

    def _net_input(self, X):
        return X @ self.weights + self.bias

    def predict(self, features):
        return np.where(self.activation(self._net_input(X=features)) >= 0.5, 1, 0)

    def activation(self, net_input):
        return net_input


if __name__ == "__main__":
    X, y = make_classification(n_features=2,
                               n_redundant=0,
                               n_informative=1,
                               n_clusters_per_class=1,
                               class_sep=1)

    alne_sgd = AdalineSGD(lr=0.01, random_state=1)
    alne_sgd.fit(X, y, epochs=15)

    print((y - alne_sgd.predict(X)))

    fig = plot_decision_regions(X=X, y=y, classifier=alne_sgd)
    plt.title("Adaline")
    plt.show()

    plt.plot(range(1, len(alne_sgd.losses) + 1), alne_sgd.losses, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Average loss")
    plt.tight_layout()
    plt.show()
