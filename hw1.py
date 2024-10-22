import numpy as np
import cv2


def conv(input,m,n,filter,bias,padding):
    #=========================== padding ===========================
    height = len(input)
    width = len(input[0])
    if padding == -1:
        paddingimg = np.copy(input)
        
    elif padding == 0:
        paddingimg = np.zeros((height + 2, width + 2))
        for i in range(height + 2):
            for j in range(width + 2):
                if i == 0 or i == height + 1 or j == 0 or j == width + 1:
                    paddingimg[i][j] = 0
                else:
                    paddingimg[i][j] = input[i-1][j-1]

    elif padding == 1:
        paddingimg = np.zeros((height + 2, width + 2))
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

    return paddingimg

#===========================maxpooling===========================
def maxpooling(input):
    height = len(input)
    width = len(input[0])
    output = np.zeros((height//2, width//2))
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            output[i//2][j//2] = max(input[i][j], input[i][j+1], input[i+1][j], input[i+1][j+1])
    return output

#===========================avgpooling===========================
def avgpooling(input):
    height = len(input)
    width = len(input[0])
    output = np.zeros((height // 2, width // 2), dtype=np.float32)
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            output[i//2][j//2] = (input[i][j] + input[i][j+1] + input[i+1][j] + input[i+1][j+1]) / 4
    return output

#=========================== mian ===========================


img = cv2.imread('imgs/IMG_5552.JPG', 0)


img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)


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

for padding in range(-1,2):
    input = img
    output = conv(input,m,n,avg,bias,padding)
    cv2.imshow('avg', output)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    output = conv(input,m,n,sobel,bias,padding)
    cv2.imshow('sobel', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(input,m,n,gaussian,bias,padding)
    cv2.imshow('gaussian', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print('padding = ', padding)


output = maxpooling(input)
cv2.imshow('maxpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = avgpooling(input)
cv2.imshow('avgpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

