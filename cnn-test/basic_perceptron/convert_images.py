from PIL import Image
import numpy as gfg
import matplotlib.image as img
imageMat = img.imread('rat.png')

print("Image shape:", imageMat.shape)

if imageMat.shape[2] == 3:
    imageMat_reshape = imageMat.reshape(imageMat.shape[0], -1)
    print("Reshaping to 2D array", imageMat_reshape.shape)
else:
    imageMat_reshape = imageMat
    gfg.savetxt('rat.csv', imageMat_reshape.reshape((3,-1)), fmt="%s", header=str(imageMat_reshape.shape))
    loaded_2D_mat = gfg.genfromtxt('rat.csv', delimiter=',', dtype=None)
    print(loaded_2D_mat)

    im = Image.fromarray(loaded_2D_mat)
    im.save("rat_mtx.jpeg")