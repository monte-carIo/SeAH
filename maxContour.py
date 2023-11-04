import numpy as np
import cv2
import sys


img = cv2.imread(sys.argv[1])
scale = .5
img = cv2.resize(img, (0,0), fx = scale, fy = scale)
# img = cv2.flip(img, 1)

h_img, w_img = img.shape[:2]

img = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('img',img)

''' canny '''
canny_img = cv2.Canny(img, 10, 200)

# hard ROI
# points = np.array([[69/w_img, 1], [75/w_img, 717/h_img], [89/w_img, 711/h_img], [102/w_img, 704/h_img], [99/w_img, 710/h_img], [130/w_img, 696/h_img], [142/w_img, 688/h_img], [250/w_img, 596/h_img], [272/w_img, 566/h_img], [321/w_img, 468/h_img], [1,400/h_img], [1,1]])

points = np.array([[72/w_img, 1 ], [92/w_img, 692/h_img], 
               [98/w_img, 687/h_img], [116/w_img,681/h_img], [132/w_img, 671/h_img], [160/w_img, 667/h_img], [1, 1]])
points *= [w_img, h_img]
cv2.fillPoly(img, pts=[points.astype(int)], color=(0, 0, 0))
cv2.fillPoly(canny_img, pts=[points.astype(int)], color=(0, 0, 0))

cv2.imshow('roi',img)
cv2.imshow('canny',canny_img)

contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
lenContours = [len(c) for c in contours]
sortidx = np.argsort(lenContours)[::-1]
print(len(contours[sortidx[0]]))

new_img = np.zeros_like(canny_img)
for i in sortidx[0:3]:
    cv2.drawContours(new_img, contours[i], -1, 255, 3)
cv2.imshow('all contour', new_img)

cv2.waitKey(0)