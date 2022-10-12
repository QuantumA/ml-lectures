from typing import Callable
import numpy as np


def euclidean_distance(point_a, point_b):
    return np.sqrt(np.sum((point_a-point_b)**2, axis=1))


def most_common(lst):
    return max(set(lst), key=lst.count)


class KNN:
    def __init__(self, k_neighbors: int = 5, dist_function: Callable = euclidean_distance):
        self.k_neighbors = k_neighbors
        self.dist_function = dist_function

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        pass

    def predict(self, X_test):
        neighbors = []
        for feature in X_test:
            distances = self.dist_function(feature, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k_neighbors])

        return list(map(most_common, neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy
