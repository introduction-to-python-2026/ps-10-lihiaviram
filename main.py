from PIL import Image
from image_utils import load_image, edge_detection
import numpy as np
from skimage.filters import median
from skimage.morphology import ball

my_image = load_image('dogphoto.jpg')
clean_image = median(my_image, ball(3))
final_edges = edge_detection(my_image)
threshold_value = 60
edge_binary = final_edges > threshold_value
binary_image_data = edge_binary.astype(np.uint8) * 255
edge_image = Image.fromarray(binary_image_data )
edge_image.save('Newphoto.png')
