import faiss
import warnings

from constants import *


class DataStore(object):
    """
    Builds a Datastore object, i.e, the set of all key-value pairs constructed from the training examples.
    One training example, consists in a vector (of size `dimension`) and a label (`int`).
    The datastore has a maximal number of examples that it can store (`capacity`).
    """
    def __init__(self, capacity, strategy, dimension, rng):
        self.capacity = capacity
        self.strategy = strategy
        self.dimension = dimension
        self.rng = rng

        self.index = faiss.IndexFlatL2(self.dimension)
        self.labels = None

    @property
    def strategy(self):
        return self.__strategy

    @strategy.setter
    def strategy(self, strategy):
        if strategy in ALL_STRATEGIES:
            self.__strategy = strategy
        else:
            warnings.warn("strategy is set to random!", RuntimeWarning)
            self.__strategy = "random"

    def build(self, train_vectors, train_labels):
        """
        add vectors to `index` according to `strategy`
        :param train_vectors:
        :type train_vectors: numpy.array os shape (n_samples, dimension)
        :param train_labels:
        :type train_labels: numpy.array of shape (n_samples,)
        """
        if self.capacity <= 0:
            return

        n_train_samples = len(train_vectors)
        if n_train_samples <= self.capacity:
            self.index.add(train_vectors)
            self.labels = train_labels
            return

        if self.strategy == "random":
            selected_indices = self.rng.choice(n_train_samples, size=self.capacity, replace=False)
            selected_vectors = train_vectors[selected_indices]
            selected_labels = train_labels[selected_indices]

            self.index.add(selected_vectors)
            self.labels = selected_labels
        else:
            raise NotImplementedError(f"{self.strategy} is not implemented")

    def clear(self):
        """
        empties the datastore by reinitializing `index` and clearing  `labels`
        """
        self.index = faiss.IndexFlatL2(self.dimension)
        self.labels = None
