import numpy as np
from activations import Activations

class Perceptron:

    def __init__(self, N, alpha=0.1, epoch=10):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
        self.epochs = epoch

    def step(self, x):
        return 1 if x > 0 else 0

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        return self.step(np.dot(X, self.W))

    def fit(self, X, y):
        X = np.c_[X, np.ones((X.shape[0]))]
        self.epochs = self.epochs
        for epoch in np.arange(0, self.epochs):
            for (x, target) in zip(X, y):
                # p = self.step(np.dot(x, self.W))
                p = Activations.sigmoid(np.dot(x, self.W))
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    print(X.shape[1])
    perceptron = Perceptron(N=X.shape[1], alpha=0.1, epoch=20)
    perceptron.fit(X, y)

    print("[INFO] testando o perceptron")

    # y = np.array([[1]])
    for (x, target) in zip(X, y):
        pred = perceptron.predict(x)
        print("[INFO] data={}, ground-truth={}, pred={}".format(
            x, target[0], pred))
