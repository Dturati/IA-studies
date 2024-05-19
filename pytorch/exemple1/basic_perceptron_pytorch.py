import numpy as np
import torch

data = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
], dtype=float)

res = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
], dtype=float)
t_res = torch.from_numpy(res)
t_data = torch.from_numpy(data)

class Percepetron:
    def __init__(self, device):
        self.device = device
        self.weigth = torch.rand(4, dtype=torch.float64)
        self.weigth = self.weigth.to(self.device)
        self.alpha = 0.1
        self.epochs = 10000

    def step(self, value):
        return 1 if value > 0 else 0

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        t_X = torch.from_numpy(X).to(self.device)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
            t_X = torch.from_numpy(X).to(self.device)

        return self.step(torch.matmul(t_X, self.weigth))

    def training(self, data: np, res: np):
        data = np.c_[data, np.ones(data.shape[0])]
        data = torch.from_numpy(data)
        data = data.to(self.device)
        for epoch in range(0, self.epochs):
            for x, y in zip(data, res):
                p = self.step(torch.matmul(x, self.weigth))
                if p != y:
                    error = p - y
                    self.weigth += -self.alpha * error * x


if __name__ == '__main__':
    device = torch.device("cuda:0")
    perceptron = Percepetron(device)
    t_res = t_res.to(device)
    perceptron.training(data=data, res=t_res)

    print("Results")

    data = np.array([
        [1, 1, 0],

    ])

    res = np.array([
        [1]
    ])
    for x, y in zip(data, res):
        prediction = perceptron.predict(x)
        print("[INFO] data={}, ground-truth={}, pred={}".format(
            x, y[0], prediction))
