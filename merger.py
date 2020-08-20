import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
from matplotlib.colors import NoNorm
import cv2
import statistics 
import scipy.misc


import numpy as np
from scipy.spatial import distance
from PIL import Image


img_no=1
prediction_paths=["1_4.png","1_8.png","1_12.png","1_16.png"]

pms=[]

for path in prediction_paths:

    tmp= mpimg.imread(path)
    tmp= np.array(tmp)
    tmp= cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    pms.append(tmp)


additive = int(255/ (len( pms )))
overlay = prediction_mask = np.zeros((pms[0].shape[0], pms[0].shape[1]))
maxa = 0
for it in range(0,len(pms)):

    overlay = np.add(overlay,pms[it])

overlay = np.divide(overlay,len(pms)) 

plt.figure(1)
plt.imshow(overlay , cmap="gray")



output = "merged"+str(img_no)+".png"

mpimg.imsave(output, overlay , cmap="gray")

plt.show()




