import numpy as np
from activations import Activations
data = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
])

res = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
])


# def cuda_avaliable() -> bool:
#     return torch.cuda.is_available()


class Percepetron:
    def __init__(self):
        self.weigth = np.random.uniform(0, 1, 4)
        self.alpha = 0.1
        self.epochs = 1000

    def step(self, value):
        return 1 if value > .5 else 0

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        return self.step(np.dot(X, self.weigth))

    def training(self, data: np, res: np):
        data = np.c_[data, np.ones(data.shape[0])]
        for epoch in range(0, self.epochs):
            for x, y in zip(data, res):
                # act = Activations.sigmoid(np.dot(x, self.weigth))
                p = self.step(np.dot(x, self.weigth))
                # p = self.step(act)
                if p != y:
                    error = p - y
                    self.weigth += -self.alpha * error * x


if __name__ == '__main__':
    perceptron = Percepetron()
    # print(perceptron.weigth)
    # print(cuda_avaliable())
    perceptron.training(data=data, res=res)

    print("Results")

    data = np.array([
        [0, 1, 0],

    ])

    res = np.array([
        [0]
    ])
    for x, y in zip(data, res):
        prediction = perceptron.predict(x)
        print("[INFO] data={}, ground-truth={}, pred={}".format(
            x, y[0], prediction))