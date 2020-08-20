import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
from matplotlib.colors import NoNorm
import cv2
import xlwt 
from xlwt import Workbook 

import numpy as np
from scipy.spatial import distance


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

    TPR = TP/(TP+FN)

    FPR = FP/(FP+TN)

    return f1_score,overlay,TPR,FPR


def sasta_diffuse(length,result):


    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
                    

            block = result[ max(0,i-int(length/2) ): min(i+int(length/2) ,result.shape[0]), max(0,j-int(length/2) ): min(j+int(length/2) ,result.shape[1])]

            result[i][j] = np.mean(block)

    return result

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
img_nos = [25]
img_paths = []

for img_no in img_nos:
    # Workbook is created 
    wb = Workbook() 
    
    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('Sheet 1') 
    for i in range(1,31):
        name = "./image_"+str(img_no)+"/"+str(img_no)+"_"+str(i)+".png"
        img_paths.append(name)

    mask_path = "./Dataset/im" + str(img_no) + "_t_mask.bmp"
    mask= mpimg.imread(mask_path)
    mask= np.array(mask)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_gray = np.divide(mask_gray,255)

    counter = 0
    THS=[0,1,2,3,4]
    col=[0,1,2,3,4,5,6,7,8,9]
    i=0
    for img_path in img_paths:
            

        
        img = mpimg.imread(img_path)
        img = np.array(img)
        
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        

        img_gray = Pass_filter(img_gray,0.55,0.55)

        score,overlay,TPR,FPR = f1_score(mask_gray,img_gray)

        print("i = ", i)
        sheet1.write(((counter//5+1)*5),col[i%10],score*100)
        i=i+1 
        print("i = ", i)
        sheet1.write(((counter//5+1)*5),col[i%10],FPR*100)
        i=i+1
        #print ("block size", (counter//5+1)*5)
        #print ("TH ", THS[counter%5])

        #print("   f1_score:",score*100)
        #print("   FPR",FPR*100,"\n\n")
        counter+=1
    out = "record_"+str(img_no)+".xls"
    wb.save(out) 

plt.imshow(overlay)
plt.show()