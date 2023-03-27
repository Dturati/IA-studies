import numpy as np
from numpy import ndarray


class CreateMatriz:
    """
        create matrix 4 by 4 with 16 elements
        :param bx, by
        :return a
    """
    def create(self, bx: int=4, by:int =4) -> ndarray:
        a = np.arange(bx * by).reshape(bx, by)
        return a


cmtx = CreateMatriz()
mtx: ndarray = cmtx.create()
print(mtx)
print(mtx.shape)