import math


class Activations:
    @staticmethod
    def sigmoid(x: float):
        return 1 / (1 + math.e ** -x)


print(Activations.sigmoid(-1))