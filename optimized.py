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


def new_chunck_remover( chunk_length, chunk_mean_threshold, distance_threshold,overlay):

    chunk_mean_array = []
    chunk_mean_max = 0 
    chunk_mean_min = 100000

    copy_overlay = overlay


    for i in range(0,overlay.shape[0]):
        for j in range(0,overlay.shape[1]):

            block = overlay[i:i+chunk_length, j:j+chunk_length]
            mean = np.mean(block)
            chunk_mean_array.append(mean)

            x_array=[]
            y_array=[]
            cx = i+chunk_length/2
            cy = j+chunk_length/2
            for x in range(0,block.shape[0]):
                for y in range(0,block.shape[1]):
                    if block[x][y] == 1:
                        x_array.append(x+i)
                        y_array.append(y+j)

            diffx = abs(cx-np.mean(x_array)) / (chunk_length/1.41)
            diffy = abs(cy-np.mean(y_array)) / (chunk_length/1.41)

            

            if mean <= chunk_mean_threshold and diffx < distance_threshold and diffy < distance_threshold:
                copy_overlay[i:i+chunk_length, j:j+chunk_length] = 0
    
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

                result[ max(0,i-int(length) ): min(i+int(length) ,result.shape[0]), max(0,j-int(length) ): min(j+int(length) ,result.shape[1])] += value*(factor**i)

                result[ max(0,i-int(length-1) ): min(i+int(length-1) ,result.shape[0]), max(0,j-int(length-1) ): min(j+int(length-1) ,result.shape[1])] -= (value*factor**i)



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
    #np.multiply(2,  np.subtract(0.5,  np.absolute( np.subtract(mask,prediction) )))

    overlay = np.multiply(2,  np.subtract(0.5,  np.absolute( np.subtract(mask,prediction) )))

    plt.figure(11)
    plt.imshow(overlay)

    return np.mean(overlay) 

def mean_sq_diff(mask,prediction):

    overlay = np.power(np.subtract(mask,prediction),2)
    summation = np.sum(overlay)

    index = (np.sum( np.power( mask,2 ) ) * np.sum( np.power( prediction,2 ) ))**0.5

    return 1-summation/index

def cross_correlation(mask,prediction):

    overlay = np.power(np.multiply(mask,prediction),2)

    summation = np.sum(overlay)

    index = (np.sum( np.power( mask,2 ) ) * np.sum( np.power( prediction,2 ) ))**0.5

    return summation/index
    

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

def Pass_filter(result,low,high):
    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
            value = result[i][j]
            if value > high:
                result[i][j] = 1
            elif value < low:
                result[i][j] = 0
    
    return result

def f1_score(mask,prediction):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    overlay = np.zeros((mask.shape[0], mask.shape[1],3))

    for i in range(0, prediction.shape[0]):
        for j in range(0, prediction.shape[1]):
            
            if prediction[i][j] == mask[i][j]:
                if prediction[i][j] == 1:
                    overlay[i][j] = [1,1,1]
                    TP+=1
                else:
                    overlay[i][j] = [0,0,0]
                    TN+=1
            else:
                if prediction[i][j] == 1:
                    overlay[i][j] = [1,0,0]
                    FP+=1
                else:
                    overlay[i][j] = [0,1,0]
                    FN+=1



    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*precision*recall/(precision+recall)

    
    return f1_score,overlay






images = [1]

for img_no in images:

    img_path = "CMF_81.png"
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

    
    img_gray = sasta_diffuse(10,img_gray)
    img_gray = Pass_filter(img_gray,0.15,0.15)
    

    
    img_gray = sasta_diffuse(10,img_gray)
    img_gray = sasta_diffuse(10,img_gray)
    img_gray = sasta_diffuse(10,img_gray)
    

    plt.figure(3)
    plt.imshow(img_gray)

    img_gray = Pass_filter(img_gray,0.25,0.25)
    

    score,overlay = f1_score(mask_gray,img_gray)

    plt.figure(2)
    plt.imshow(overlay)
    plt.show()

    
    print ("img_no ",img_no)

    
    print("   cross correlation:",cross_correlation( mask_gray,img_gray))
    print("   f1_score:",score)

  

    



















