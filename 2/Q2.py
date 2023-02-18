import math

import cv2
import numpy as np

img = cv2.imread('./Q1-09-edge.jpg')
edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

d = int(math.sqrt(edges.shape[0] ** 2 + edges.shape[1] ** 2))

# binary image
edges = np.vectorize(lambda x: 1 if x > 250 else 0)(edges)

# new_edges = np.ones(edges.shape)

# for i in range(5, edges.shape[0] - 5):
#     for j in range(5, edges.shape[1] - 5):
#         box = edges[i - 5:i + 5, j - 5:j + 5]
#         if np.sum(box) > 20 or np.sum(box) < 4:
#             edges[i, j] = 0  # or can be new_edges ! but may be it is better
#
# edges = np.vectorize(lambda x: 255 if x == 1 else 0)(edges)
# cv2.imwrite("out/edges.jpg", edges)

edges = cv2.imread("out/edges.jpg")
edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
edges = np.vectorize(lambda x: 255 if x > 250 else 0)(edges)

# p -> (-d,d) theta -> (-180,180)
p_theta = np.zeros((2 * d, 2 * 180))

offset = d


def fillPThetaSpace(x, y, pThetaArray, offset):
    for theta in range(-90, 91, 1):
        p = x * math.cos(np.deg2rad(theta)) + y * math.sin(np.deg2rad(theta))
        pThetaArray[offset + int(p)][theta + 90] += 1


def houghAlgo(image, pThetaArray):
    Y, X = np.nonzero(image)
    for i, j in zip(X, Y):
        fillPThetaSpace(i, j, pThetaArray, offset)
    return pThetaArray


p_theta = houghAlgo(edges, p_theta)

# p_theta = p_theta / p_theta.max() * 255
# p_theta = np.asarray(p_theta, 'uint8')
# cv2.imwrite("out/p_theta.jpg", p_theta)



