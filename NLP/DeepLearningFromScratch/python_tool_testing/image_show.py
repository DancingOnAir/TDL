import matplotlib.pyplot as plt
from matplotlib.image import imread


img = imread('../dataset/lena.png')

# imshow()接收一张图像，只是画出该图，并不会立刻显示出来。
# imshow后还可以进行其他draw操作，比如scatter散点等。
plt.imshow(img)
plt.show()
