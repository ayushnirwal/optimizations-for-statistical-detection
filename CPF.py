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



def CPF(img_gray,mask_gray,block_size,mean_threshold,SD_threshold,min_pixel_distance,check_offset,SD_TH):
    
    


    column = img_gray.shape[1] - block_size +1
    row = img_gray.shape[0] - block_size +1

    prediction_mask = np.zeros((img_gray.shape[0], img_gray.shape[1]))


    data = []


    #creating array of dict objects ( with mean, SD, i, j ) by calc mean and SD of every block

    #print("making blocks and calculating mean with block_size = " + str(block_size))

    block_counter = 0
    
    for i in range(0, row):
        for j in range(0, column):

            block = img_gray[i:i+block_size, j:j+block_size]


            d=dict()
            d['M'] = np.mean(block)
            d['SD'] = np.std(block)
            d['i'] =    i
            d['j'] =    j
            


            data.append(d)

            block_counter+=1
            
            
    #print("Done")
    

    # sorting according to Mean
    sorted_mean = sorted(data, key=lambda element: element['M']) 



    # distinguishing similar blocks ( only checking neighbours in sorted array for potential similar blocks) 
    sim_array=[]

    for i in range(len(sorted_mean)):

        for j in range( max(0,i-check_offset), min( len(sorted_mean), i+check_offset) ):
            mean_similarity = abs(sorted_mean[j]['M'] - sorted_mean [i]['M'])
            SD_similarity = abs(sorted_mean[j]['SD'] - sorted_mean [i]['SD'])

            coor1 = np.array([ sorted_mean[i]['i'] , sorted_mean[i]['j']])
            coor2 = np.array([ sorted_mean[j]['i'] , sorted_mean[j]['j']])

            distance = np.linalg.norm(coor1-coor2)

            if SD_TH == None:

                if mean_similarity <= mean_threshold and SD_similarity <= SD_threshold and distance >= min_pixel_distance:
                    
                    sim_array.append(sorted_mean[i])
                    sim_array.append(sorted_mean[j])
            else:

                if mean_similarity <= mean_threshold and SD_similarity <= SD_threshold and distance >= min_pixel_distance and  abs( sorted_mean[j]['SD'] >= SD_TH) :
                    
                    sim_array.append(sorted_mean[i])
                    sim_array.append(sorted_mean[j])


    #creating prediction mask from similar blocks

    for ele in sim_array:
        i = ele['i']
        j = ele['j']
        prediction_mask [i:i+block_size, j:j+block_size] = 255
    
    return prediction_mask

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

def correction(result,length,thres):

    print(" the new thing with length:",length)
    print(" with threshold",thres)

    corrected = np.zeros((result.shape[0], result.shape[1]))
    
    for i in range(length,result.shape[0]-length):
            for j in range(length,result.shape[1]-length):

                block = result[ i: min( i+length, result.shape[0]), min( j+length, result.shape[1])]

                
                if int( np.mean(block) ) > thres:

                    corrected[ i: min( i+length, result.shape[0]), min( j+length, result.shape[1])] = int( np.mean(block) )

                else:
                    
                    corrected[i][j] = 0

    return corrected

def compare_result(prediction_mask,overlay,thres):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    overlay = np.zeros((overlay.shape[0], overlay.shape[1],3))

    for i in range(0, prediction_mask.shape[0]):
        for j in range(0, prediction_mask.shape[1]):
            if prediction_mask [i][j] >= thres:
                prediction_mask[i][j] == 255
            if prediction_mask[i][j] == mask_gray[i][j]:
                if prediction_mask[i][j] == 255:
                    overlay[i][j] = [1,1,1]
                    TP+=1
                else:
                    overlay[i][j] = [0,0,0]
                    TN+=1
            else:
                if prediction_mask[i][j] == 255:
                    overlay[i][j] = [1,0,0]
                    FP+=1
                else:
                    overlay[i][j] = [0,1,0]
                    FN+=1

    # if TP+FN == 0:
    #     return overlay,0

    # precision = TP/(TP+FP)
    # recall = TP/(TP+FN)
    # f1_score = 2*precision*recall/(precision+recall)

    
    return overlay

def accuracy(prediction, mask ,high):

    acc = []
    for i in range( 0, prediction.shape[0]):
        for j in range( 0, prediction.shape[1]):

            diff = prediction[i][j] - mask[i][j]

            (high-diff) / high

            print( diff)

            acc.append(diff)

    return ( 1 - statistics.mean( acc ) )











#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

pre_comp = False
mean_threshold = 0
SD_threshold = 0
min_pixel_distance = 100
check_offset = 5

for img_no in [22,23,24,25]:
    print("image_no :",img_no)
    
    img_path = "./Dataset/im" + str(img_no) + "_t.bmp"
    mask_path = "./Dataset/im" + str(img_no) + "_t_mask.bmp"


    img = mpimg.imread(img_path)
    img = np.array(img)

    mask= mpimg.imread(mask_path)
    mask= np.array(mask)

    #converting to grayscale

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    block_sizes =[5,10,15,20,25,30]
    SD_THs = [None,1,2,3,4]
    pms = []
    counter = 1
    for size in block_sizes:
        for SD_TH in SD_THs:
            
            print("size :",size," SD_TH:",SD_TH)
            
            pm = CPF( img_gray,mask_gray,size,mean_threshold,SD_threshold,min_pixel_distance,check_offset,SD_TH)

            output = str(img_no)+"_"+str(counter)+".png"

            mpimg.imsave(output, pm , cmap="gray")

            counter+=1
            

    

    

    
                



    

    


    

    
    



    

    
plt.show()
    
    
    


