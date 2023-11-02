
import cv2
import numpy as np
import sys
MIN_ANGLE_DIFF = 12
W_SIZE = 35
MIN_DENSITY=0.07
DISTANCE = 10
from edge_searching_psuedo import *

def houghLines(img):
        return cv2.HoughLinesP(img, 1, np.pi/180, 10, minLineLength=10, maxLineGap=3)

def findStartingPoint(mask_img, canny_img, img_draw):

    startX, startY = .35 , .90 #TODO - not fixed but auto detect starting point
    h_img, w_img = img.shape[:2]
    x1, y1 = int(startX*w_img -W_SIZE/2), int(startY * h_img + W_SIZE/2)
    x2, y2 = x1+W_SIZE, y1-W_SIZE
    for i in range(10):
        w_mask=  mask_img[y2:y1, x1:x2]
        w_edge=  canny_img[y2:y1, x1:x2]
        HTsegments = houghLines(w_edge)
        if HTsegments is None:
            print('cannot find HT')
            y1 = y1-W_SIZE//2
            y2 = y2-W_SIZE//2
            continue
        HTsegments = HTsegments.reshape(-1,4)
        angles = np.arctan2(-(HTsegments[:,3]-HTsegments[:,1]), HTsegments[:,2]-  HTsegments[:,0]) / np.pi * 180 
        for i in range(len(angles)):
            if angles[i] < 0:
                angles[i] +=180
        print('angle', angles)
        angles = angles[angles <36]
        

        cv2.imshow('w_mask', w_mask)
        cv2.imshow('w_edge', w_edge)
        drawSegments(HTsegments, prompt='hough')
        
        targetP=medianPoint(w_mask)
        
        cv2.waitKey(0)
        if targetP != None and len(angles): # NOTE: still need stricter conditions 
            cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,255,255))
            cv2.imshow('img', img_draw)
            cv2.waitKey(0)
            break
            # cv2.rectangle(mask_img_draw, (x1,y1), (x2, y2), (200,200,200), thickness=2)
        else:
            
            cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,0,0), thickness=2)
            cv2.imshow('img', img_draw)
            # cv2.rectangle(mask_img_draw, (x1,y1), (x2, y2), (100,100,100), thickness= 2)
        y1 = y1-W_SIZE//2
        y2 = y2-W_SIZE//2
    return x1, y1, x2, y2


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    startX, startY = .35 , .90 #TODO - not fixed but auto detect starting point
    scale = .5
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)

    h_img, w_img = img.shape[:2]

    # hard ROI
    img = cv2.bilateralFilter(img, 9, 75, 75)

    ''' canny '''
    canny_img = cv2.Canny(img, 50, 200)

    ''' hsv seg'''
    mask_img = hsv_mask(img).astype(np.uint8)

    ''' strict ROI '''
    points = np.array([[72/w_img, 1 ], [92/w_img, 692/h_img], 
                   [98/w_img, 687/h_img], [116/w_img,681/h_img], [132/w_img, 671/h_img], [160/w_img, 667/h_img], [1, 1]])
    points *= [w_img, h_img]
    cv2.fillPoly(img, pts=[points.astype(int)], color=(0, 0, 0))
    cv2.imshow('roi',img)
    cv2.fillPoly(canny_img, pts=[points.astype(int)], color=(0, 0, 0))
    cv2.fillPoly(mask_img, pts=[points.astype(int)], color=(0, 0, 0))
    cv2.imshow('canny',canny_img)
    cv2.imshow('roi',img)
    mask_img_draw = mask_img.copy()
    img_draw = img.copy()
    print(findStartingPoint(mask_img, canny_img, img_draw))


    # inner_segs, outer_segs = sliding_window(img) #FIXME - stop without median condition 