import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('/data/generated_data/0/train/2768-0.jpg')
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()