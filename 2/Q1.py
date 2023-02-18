import numpy as np
import cv2
import scipy.ndimage
from PIL import Image
import math
from scipy import signal


img = cv2.imread("./Books.jpg" , cv2.IMREAD_UNCHANGED)
height , width , channels = img.shape

# Convert the img to grayscale 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def gaussian_derivative(x, y , sigma):
    return ((-x) / (2 * math.pi * sigma ** 4)) * math.exp(-(((x ** 2) + (y  ** 2)) / (2 * sigma ** 2)))


def guassian_derivative_kernel_x(size, sigma):
    mat = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mat[i, j] = gaussian_derivative(j- (size//2), i- (size//2), sigma)
    return mat

def guassian_derivative_kernel_y(size, sigma):
    mat = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mat[i, j] = gaussian_derivative(i- (size//2), j- (size//2), sigma)
    return mat

x = signal.convolve2d(gray, guassian_derivative_kernel_x(7, 1), 'same')
print(guassian_derivative_kernel_x(7, 1))
print("************************************")
cv2.imwrite("‫‪Q1-05-hor.jpg", x)
y = signal.convolve2d(gray, guassian_derivative_kernel_y(7, 1 ), 'same')
print(guassian_derivative_kernel_x(7, 1))
print("************************************")
cv2.imwrite("‫‪Q1-06-ver.jpg‬‬", y)

result = np.hypot(x , y)
cv2.imwrite('‫‪Q1-07-grad-mag.jpg‬‬', result)

theta = np.arctan2(y , x)
cv2.imwrite('‫‪Q1-08-grad-dir.jpg‬‬' , np.asarray(theta , 'uint8'))

def threshold(x):
    if x > 10:
        return 255
    else:
        return 0

edges = result / result.max() * 255
edges = np.asarray(edges, dtype='uint8')
result = np.vectorize(threshold)(result)
cv2.imwrite("‫‪Q1-09-edge.jpg‬‬", result)


def guassian_derivative_hor_row_kernel(size, sigma):
    mat = np.zeros((1, size))
    for i in range(size):
        mat[0, i] =((-(i - (size//2)) / (math.sqrt(2 * math.pi) * sigma ** 2)) * math.exp(-((i - (size//2)) ** 2) / (2 * sigma ** 2)))
    return mat

def guassian_derivative_hor_col_kernel(size, sigma):
    mat = np.zeros((size, 1))
    for i in range(size):
        mat[i, 0] =  ((1 / (math.sqrt(2 * math.pi) * sigma ** 2)) * math.exp(-((i - (size//2)) ** 2) / (2 * sigma ** 2)))
    return mat

def guassian_derivative_ver_row_kernel(size, sigma):
    mat = np.zeros((1, size))
    for i in range(size):
        mat[0, i] =((-(i - (size//2)) / (math.sqrt(2 * math.pi) * sigma ** 2)) * math.exp(-((i - (size//2)) ** 2) / (2 * sigma ** 2)))
    return mat


def guassian_derivative_ver_col_kernel(size, sigma):
    mat = np.zeros((size, 1))
    for i in range(size):
        mat[i, 0] = ((1 / (math.sqrt(2 * math.pi) * sigma ** 2)) * math.exp(-((i - (size//2)) ** 2) / (2 * sigma ** 2)))
    return mat


row_x_img = signal.convolve2d(gray, guassian_derivative_hor_row_kernel(7, 1), 'same')
print(guassian_derivative_hor_row_kernel(7 , 1))
print("************************************")
cv2.imwrite("‫‪Q1-01-hor-row.jpg‬‬", row_x_img)

col_x_img = signal.convolve2d(row_x_img, guassian_derivative_hor_col_kernel(7, 1), 'same')
print(guassian_derivative_hor_col_kernel(7, 1))
print("************************************")
cv2.imwrite("‫‪Q1-03-hor-col.jpg‬‬", col_x_img)

row_y_img = signal.convolve2d(gray, guassian_derivative_ver_col_kernel(7, 1).reshape((1, 7)), 'same')
print(guassian_derivative_ver_col_kernel(7, 1).reshape((1, 7)))
print("************************************")
cv2.imwrite("‫‪Q1-04-ver-col.jpg‬‬", row_y_img)

col_y_img = signal.convolve2d(row_y_img, guassian_derivative_ver_row_kernel(7, 1).reshape((7,1)), 'same')
print(guassian_derivative_ver_row_kernel(7, 1).reshape((7,1)))
print("************************************")
cv2.imwrite("‫‪Q1-02-ver-row.jpg‬‬", col_y_img)

