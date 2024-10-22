import numpy as np
import cv2


def conv(input,m,n,filter,bias,padding):
    #=========================== padding ===========================
    height = len(input)
    width = len(input[0])
    if padding == -1:
        paddingimg = input
    elif padding == 0:
        for i in range(height + 2):
            for j in range(width + 2):
                if i == 0 or i == height + 1 or j == 0 or j == width + 1:
                    paddingimg[i][j] = 0
                else:
                    paddingimg[i][j] = input[i-1][j-1]

    elif padding == 1:
        for i in range(height + 2):
            for j in range(width + 2):
                if i == 0 or i == height + 1 or j == 0 or j == width + 1:
                    paddingimg[i][j] = 255
                else:
                    paddingimg[i][j] = input[i-1][j-1]    
    #=========================== convolution ===========================
    height = len(paddingimg)
    width = len(paddingimg[0])
    for i in range(height - m + 1):
        for j in range(width - n + 1):
            sum = 0
            for k in range(m):
                for l in range(n):
                    sum += paddingimg[i+k][j+l] * filter[k][l]
            paddingimg[i][j] = sum + bias

#=========================== mian ===========================

# Read the image
img = cv2.imread('imgs/IMG_5552.JPG', 0)

#  Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

input = img
m = 3
n = 3
avg = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
    ])
sobel = np.array([
    [1/9, 0 , -1/9],
    [1/9, 0 , -1/9],
    [1/9, 0 , -1/9]
    ])
gaussian = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
    ])
bias = 0
padding = -1 

