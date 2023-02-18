import cv2
import numpy as np
import math

def homography_matrix(x1, y1, x2, y2, x3, y3, x4, y4, f_x1, f_y1, f_x2, f_y2, f_x3, f_y3, f_x4, f_y4):
    H = np.array([[-x1, -y1, -1, 0, 0, 0, x1 * f_x1, y1 * f_x1, f_x1],
                  [0, 0, 0, -x1, -y1, -1, x1 * f_y1, y1 * f_y1, f_y1],

                  [-x2, -y2, -1, 0, 0, 0, x2 * f_x2, y2 * f_x2, f_x2],
                  [0, 0, 0, -x2, -y2, -1, x2 * f_y2, y2 * f_y2, f_y2],

                  [-x3, -y3, -1, 0, 0, 0, x3 * f_x3, y3 * f_x3, f_x3],
                  [0, 0, 0, -x3, -y3, -1, x3 * f_y3, y3 * f_y3, f_y3],

                  [-x4, -y4, -1, 0, 0, 0, x4 * f_x4, y4 * f_x4, f_x4],
                  [0, 0, 0, -x4, -y4, -1, x4 * f_y4, y4 * f_y4, f_y4],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]
                  ])

    zero_matrix = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
    solution = np.linalg.solve(H, zero_matrix)
    return solution


img = cv2.imread("./Books.jpg")

#edge points of book1
x1, y1, x2, y2, x3, y3, x4, y4 = 666, 214, 600, 402, 323, 292, 382, 108
#calculating the size of the book by its edge points
width_book1 = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
height_book1 = math.sqrt((y2 - y3) ** 2 + (x2 - x3) ** 2)
#calculating the homography matrix for book3
A = homography_matrix(x1, y1, x2, y2, x3, y3, x4, y4,  0, 0, width_book1, 0, width_book1, height_book1, 0, height_book1)
A = np.reshape(A, (3, 3))
final_book1 = cv2.warpPerspective(img, A , (img.shape[0], img.shape[1]))
croped_book1 = final_book1[:int(height_book1), :int(width_book1)]
cv2.imwrite("Q3_book1.jpg", croped_book1)

##########################################################
#edge points of book2
x1, y1, x2, y2, x3, y3, x4, y4 = 220 , 430 , 400 , 470 , 370 , 730 ,145 , 710
#calculating the size of the book by its edge points
width_book2 = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
height_book2 = math.sqrt((y2 - y3) ** 2 + (x2 - x3) ** 2)
#calculating the homography matrix for book3
B = homography_matrix(x1, y1, x2, y2, x3, y3, x4, y4,  0, 0, width_book2, 0, width_book2, height_book2, 0, height_book2)
B = np.reshape(B, (3, 3))
final_book2 = cv2.warpPerspective(img, B , (img.shape[0], img.shape[1]))
croped_book2 = final_book2[:int(height_book2), :int(width_book2)]
cv2.imwrite("Q3_book2.jpg", croped_book2)

############################################################
#edge points of book3
x1, y1, x2, y2, x3, y3, x4, y4 = 816, 972, 611, 1105, 425, 800, 625, 665
#calculating the size of the book by its edge points
width_book3 = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
height_book3 = math.sqrt((y2 - y3) ** 2 + (x2 - x3) ** 2)
#calculating the homography matrix for book3
C = homography_matrix(x1, y1, x2, y2, x3, y3, x4, y4,  0, 0, width_book3, 0, width_book3, height_book3, 0, height_book3)
C = np.reshape(C, (3, 3))
final_book3 = cv2.warpPerspective(img, C , (img.shape[0], img.shape[1]))
croped_book3 = final_book3[:int(height_book3), :int(width_book3)]
cv2.imwrite("Q3_book3.jpg", croped_book3)
