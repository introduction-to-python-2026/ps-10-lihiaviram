from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import ball

def load_image(path):
   image=Image.open('dogphoto.jpg')
   image=np.array(image)
    return image

def edge_detection(image):
    image_gray = np.mean(image, axis=2)
    plt.imshow(image_gray, cmap='gray');
    clean_image = median(image_gray, ball(3))
    kernelY= np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    kernelX= np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    edgeX = convolve2d(image, kernelX, mode='same', boundary='symm')
    edgeY = convolve2d(image, kernelY, mode='same', boundary='symm')
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
