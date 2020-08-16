import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
from matplotlib.colors import NoNorm
import cv2
from PIL import Image

img_no = 25

img_path = "output1.png"
mask_path = "./Dataset/im" + str(img_no) + "_t_mask.bmp"


img = mpimg.imread(img_path)
img = np.array(img)

mask= mpimg.imread(mask_path)
mask= np.array(mask)

mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

mask_gray = np.divide(mask_gray,255)

plt.figure(1)
plt.imshow(img_gray)

plt.figure(2)
plt.imshow(mask_gray)

op1 = np.subtract(0.5,np.absolute(np.subtract(mask_gray,img_gray)))

op2 = np.subtract(1,np.absolute(np.subtract(mask_gray,img_gray)))

plt.figure(3)
plt.imshow(op1)

plt.figure(4)
plt.imshow(op2)





plt.show()