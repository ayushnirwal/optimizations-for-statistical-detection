import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
from matplotlib.colors import NoNorm
import cv2


import numpy as np
from scipy.spatial import distance

#initializing parameters

block_size = 10
pre_comp = False
mean_threshold = 0
SD_threshold = 0
min_pixel_distance = 100
check_offset = 2


#reading image and provided mask ( provided mask will be used for accuracy calulation)

img = mpimg.imread('./Dataset/im2_t.bmp')
img = np.array(img)

mask= mpimg.imread('./Dataset/im2_t_mask.bmp')
mask= np.array(mask)

#converting to grayscale

mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

if pre_comp == True:
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2

    img_gray=LL

    coeffs2 = pywt.dwt2(mask_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2

    mask_gray=LL


column = img_gray.shape[1] - block_size + 1
row = img_gray.shape[0] - block_size + 1

prediction_mask = np.zeros((img_gray.shape[0], img_gray.shape[1]))


data = []


#creating array of dict objects ( with mean, SD, i, j ) by calc mean and SD of every block

print("making blocks and calculating mean with block_size = " + str(block_size))

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
        
        
print("Done")

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

        if mean_similarity <= mean_threshold and SD_similarity <= SD_threshold and distance >= min_pixel_distance:
            
            sim_array.append(sorted_mean[i])
            sim_array.append(sorted_mean[j])


print(len(sim_array))

#creating prediction mask from similar blocks

for ele in sim_array:
    i = ele['i']
    j = ele['j']
    prediction_mask [i:i+block_size, j:j+block_size] = 255


plt.figure(1)
plt.imshow(prediction_mask,cmap="gray")

pm1=prediction_mask

#=======================================================================================================================================

block_size = 20
pre_comp = False
mean_threshold = 0
SD_threshold = 0
min_pixel_distance = 100
check_offset = 2



if pre_comp == True:
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2

    img_gray=LL

    coeffs2 = pywt.dwt2(mask_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2

    mask_gray=LL


column = img_gray.shape[1] - block_size + 1
row = img_gray.shape[0] - block_size + 1

prediction_mask = np.zeros((img_gray.shape[0], img_gray.shape[1]))


data = []


#creating array of dict objects ( with mean, SD, i, j ) by calc mean and SD of every block

print("making blocks and calculating mean with block_size = " + str(block_size))

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
        
        
print("Done")

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

        if mean_similarity <= mean_threshold and SD_similarity <= SD_threshold and distance >= min_pixel_distance:
            
            sim_array.append(sorted_mean[i])
            sim_array.append(sorted_mean[j])


print(len(sim_array))

#creating prediction mask from similar blocks

for ele in sim_array:
    i = ele['i']
    j = ele['j']
    prediction_mask [i:i+block_size, j:j+block_size] = 255


plt.figure(2)
plt.imshow(prediction_mask,cmap="gray")

#===========================================================================================================================

pm2 =prediction_mask


#=======================================================================================================================================

block_size = 5
pre_comp = False
mean_threshold = 0
SD_threshold = 0
min_pixel_distance = 100
check_offset = 2



if pre_comp == True:
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2

    img_gray=LL

    coeffs2 = pywt.dwt2(mask_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2

    mask_gray=LL


column = img_gray.shape[1] - block_size + 1
row = img_gray.shape[0] - block_size + 1

prediction_mask = np.zeros((img_gray.shape[0], img_gray.shape[1]))


data = []


#creating array of dict objects ( with mean, SD, i, j ) by calc mean and SD of every block

print("making blocks and calculating mean with block_size = " + str(block_size))

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
        
        
print("Done")

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

        if mean_similarity <= mean_threshold and SD_similarity <= SD_threshold and distance >= min_pixel_distance:
            
            sim_array.append(sorted_mean[i])
            sim_array.append(sorted_mean[j])


print(len(sim_array))

#creating prediction mask from similar blocks

for ele in sim_array:
    i = ele['i']
    j = ele['j']
    prediction_mask [i:i+block_size, j:j+block_size] = 255


plt.figure(3)
plt.imshow(prediction_mask,cmap="gray")

pm3 = prediction_mask

overlay=np.zeros((img_gray.shape[0], img_gray.shape[1]))

for i in range(0,prediction_mask.shape[0]):
    for j in range(0,prediction_mask.shape[1]):
        if pm1[i][j] == 255 and pm2[i][j] == 255 and pm3[i][j] == 255:
            overlay[i][j] = 255
        


plt.figure(4)
plt.imshow(overlay,cmap="gray")

plt.show()




#===========================================================================================================================

# chunk_length = 20
# chunk_mean_max = 0
# chunk_mean_min = 4000
# chunk_mean_threshold = 120
# chunk_mean_array = []

# for i in range(0,prediction_mask.shape[0],chunk_length):
#     for j in range(0,prediction_mask.shape[1],chunk_length):

#         block = prediction_mask[i:i+chunk_length, j:j+chunk_length]
#         mean = np.mean(block)
#         chunk_mean_array.append(mean)
#         if mean > chunk_mean_max:
#             chunk_mean_max = mean
#         if mean < chunk_mean_min:
#             chunk_mean_min = mean

#         if mean <= chunk_mean_threshold:
#             prediction_mask [i:i+chunk_length, j:j+chunk_length] = 0
        
# plt.figure(7)
# plt.hist(chunk_mean_array, bins=50)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
        



#creating overlay for visual representation of comparision b/w predicted mask and provided mask
#black => True negative
#white => True positive
#green => False positive
#red => False positive


# overlay=np.zeros((img_gray.shape[0], img_gray.shape[1],3))


# TP = 0
# FP = 0
# TN = 0
# FN = 0

# for i in range(0, prediction_mask.shape[0]):
#     for j in range(0, prediction_mask.shape[1]):
#         if prediction_mask[i][j] == mask_gray[i][j]:
#             if prediction_mask[i][j] == 255:
#                 overlay[i][j] = [255,255,255]
#                 TP+=1
#             else:
#                 overlay[i][j] = [0,0,0]
#                 TN+=1
#         else:
#             if prediction_mask[i][j] == 255:
#                 overlay[i][j] = [255,0,0]
#                 FP+=1
#             else:
#                 overlay[i][j] = [0,255,0]
#                 FN+=1

# plt.figure(3)
# plt.imshow(prediction_mask,cmap="gray")

# plt.figure(4)
# plt.imshow(overlay,cmap="gray")


# plt.show()



# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# accuracy = 2*precision*recall/(precision+recall)
# print("accuracy:",accuracy)















