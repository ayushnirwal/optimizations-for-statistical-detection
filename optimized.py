import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
from matplotlib.colors import NoNorm
import cv2


import numpy as np
from scipy.spatial import distance







def chunck_remover( chunk_length, chunk_mean_threshold, overlay):

    chunk_mean_array = []
    chunk_mean_max = 0 
    chunk_mean_min = 100000


    for i in range(0,overlay.shape[0],chunk_length):
        for j in range(0,overlay.shape[1],chunk_length):

            block = overlay[i:i+chunk_length, j:j+chunk_length]
            mean = np.mean(block)
            chunk_mean_array.append(mean)
            if mean > chunk_mean_max:
                chunk_mean_max = mean
            if mean < chunk_mean_min:
                chunk_mean_min = mean

            if mean <= chunk_mean_threshold:
                overlay [i:i+chunk_length, j:j+chunk_length] = 0
    
    return overlay


def correction(length,thres,result):

    corrected = np.zeros((result.shape[0], result.shape[1]))
    
    for i in range(length,result.shape[0]-length):
            for j in range(length,result.shape[1]-length):

                block = result[ i: min( i+length, result.shape[0]), min( j+length, result.shape[1])]

                
                if int( np.mean(block) ) > thres:

                    corrected[ i: min( i+length, result.shape[0]), min( j+length, result.shape[1])] = 1

                else:
                    
                    corrected[i][j] = 0

    return corrected

def sasta_diffuse(length,result):


    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
                    

            block = result[ max(0,i-int(length/2) ): min(i+int(length/2) ,result.shape[0]), max(0,j-int(length/2) ): min(j+int(length/2) ,result.shape[1])]

            result[i][j] = np.mean(block)

    return result

def diffuse(length,factor,result):


    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
                    
            for it in range(length):
                value = result[i][j]

                result[ max(0,i-int(length) ): min(i+int(length) ,result.shape[0]), max(0,j-int(length) ): min(j+int(length) ,result.shape[1])] += value*factor

                result[ max(0,i-int(length-1) ): min(i+int(length-1) ,result.shape[0]), max(0,j-int(length-1) ): min(j+int(length-1) ,result.shape[1])] -= value*factor



    return result

def normalize(result):

    high = np.amax(result)
    low = np.amin(result)

    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
            value = result[i][j]
            result[i][j] = ( value - low ) / (high - low)

    return result

def cal_accuracy(mask,prediction):

    overlay = np.multiply(2,  np.subtract(0.5,  np.absolute( np.subtract(mask,prediction) )))


    return np.mean(overlay) 

def to_one(result,threshold):

    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
            value = result[i][j]
            if value > threshold:
                result[i][j] = 1
    
    return result


def to_zero(result,threshold):

    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
            value = result[i][j]
            if value < threshold:
                result[i][j] = 0
    
    return result



img_path = "name.png"


img_no = 25

img_path = "name.png"
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

print(cal_accuracy(mask_gray,img_gray))


result = chunck_remover(int(img_gray.shape[0]*0.03),0.2,img_gray)

plt.figure(2)
plt.imshow(result)

print( cal_accuracy(mask_gray,result) )

result = chunck_remover(int(result.shape[0]*0.05),0.2,result)

result = chunck_remover(int(result.shape[0]*0.07),0.2,result)

result = chunck_remover(int(result.shape[0]*0.1),0.2,result)

result = chunck_remover(int(result.shape[0]*0.05),0.2,result)

result = chunck_remover(int(result.shape[0]*0.07),0.2,result)

result = chunck_remover(int(result.shape[0]*0.1),0.2,result)

result = chunck_remover(int(result.shape[0]*0.05),0.2,result)

result = chunck_remover(int(result.shape[0]*0.07),0.2,result)

result = chunck_remover(int(result.shape[0]*0.1),0.2,result)

print( cal_accuracy(mask_gray,result) )

plt.figure(3)
plt.imshow(result)

result = diffuse(5,0.01,result)
result = normalize(result)
print( cal_accuracy(mask_gray,result) )

result = to_zero(result,0.35)
print( cal_accuracy(mask_gray,result) )

result = to_one(result,0.7)

result = diffuse(5,0.01,result)
result = normalize(result)
print( cal_accuracy(mask_gray,result) )

result = to_zero(result,0.35)
print( cal_accuracy(mask_gray,result) )

result = to_one(result,0.7)

print( cal_accuracy(mask_gray,result) )

plt.figure(4)
plt.imshow(result)




plt.show()

