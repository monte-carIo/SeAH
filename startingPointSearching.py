import cv2
import numpy as np
import sys

def findStartingPoint(mask_img, canny_img, img_draw, minA=0, maxA=32, startX=.35, startY = .92):
    h_img, w_img = img_draw.shape[:2]
    x1, y1 = int(startX*w_img -W_SIZE/2), int(startY * h_img + W_SIZE/2)
    x2, y2 = x1+W_SIZE, y1-W_SIZE
    cv2.imshow('mask', mask_img)
    num_reg =  6
    vote=[(0,())]*num_reg
    for i in range(num_reg):
        print('i',i)
        w_mask=  mask_img[y2:y1, x1:x2]
        w_edge=  canny_img[y2:y1, x1:x2]
        cv2.imshow('w_mask', w_mask)
        cv2.imshow('w_edge', w_edge)
        HTsegments = houghLines(w_edge, vote = 10, minl=15)
        if HTsegments is None:
            print('cannot find HT')
            y1 = y1-W_SIZE//2
            y2 = y2-W_SIZE//2
            minA +=5
            maxA +=5
            vote[i] = 0, ()
            continue
        HTsegments = HTsegments.reshape(-1,4)
        angles = np.arctan2(-(HTsegments[:,3]-HTsegments[:,1]), HTsegments[:,2]-  HTsegments[:,0]) / np.pi * 180 
        for j in range(len(angles)):
            if angles[j] < 0:
                angles[j] +=180
        print('angle', angles)
        # angles = angles[angles <36]
        fil = np.where((angles < maxA) & (angles > minA))
        angles = angles[fil]
        print('min', minA)
        print('max', maxA)

        drawSegments(HTsegments, prompt='hough')
        
        targetP, density =medianPoint(w_mask)
        
        # cv2.waitKey(0)
        if targetP != None and density > AVERAGE_DENSITY and len(angles): # NOTE: still need stricter conditions 
            cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,255,255))
            cv2.imshow('img', img_draw)
            cv2.waitKey(0)
            print(x1, y1, x2, y2, np.mean(angles))
            print('add', i)
            vote[i] = len(angles), (x1, y1, x2, y2, np.mean(angles))
            # return x1, y1, x2, y2, np.mean(angles)
            # cv2.rectangle(mask_img_draw, (x1,y1), (x2, y2), (200,200,200), thickness=2)
        else:
            
            vote[i] = 0, ()
            cv2.rectangle(img_draw, (x1,y1), (x2, y2), (255,0,0), thickness=2)
            cv2.imshow('img', img_draw)
            pass
            # cv2.rectangle(mask_img_draw, (x1,y1), (x2, y2), (100,100,100), thickness= 2)
        y1 = y1-W_SIZE//2
        y2 = y2-W_SIZE//2
        minA +=5
        maxA +=5
        cv2.waitKey(0)
    scores = [s[0] for s in vote]
    idMax = np.argmax(scores)
    return vote[idMax][1]

if __name__ == '__main__':
    from edge_searching_psuedo import hsv_mask, W_SIZE, houghLines, drawSegments, medianPoint, AVERAGE_DENSITY
    img = cv2.imread(sys.argv[1])
    startX, startY = .35 , .95 #TODO - not fixed but auto detect starting point
    scale = .5
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)
    img = cv2.flip(img, 1)

    h_img, w_img = img.shape[:2]

    # hard ROI
    img = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow('img',img)

    ''' canny '''
    canny_img = cv2.Canny(img, 30, 200)

    ''' hsv seg'''
    mask_img = hsv_mask(img, lower=(0,0,40), upper=(110,100,150)).astype(np.uint8)

    ''' strict ROI '''
    
    points = np.array([[69/w_img, 1], [75/w_img, 717/h_img], [89/w_img, 711/h_img], [102/w_img, 704/h_img], [99/w_img, 710/h_img], [130/w_img, 696/h_img], [142/w_img, 688/h_img], [230/w_img, 632/h_img], [290/w_img, 566/h_img], [321/w_img, 468/h_img], [1,400/h_img], [1,1]])

    # points = np.array([[72/w_img, 1 ], [92/w_img, 692/h_img], 
    #                [98/w_img, 687/h_img], [116/w_img,681/h_img], [132/w_img, 671/h_img], [160/w_img, 667/h_img], [1, 1]])
    points *= [w_img, h_img]
    cv2.fillPoly(img, pts=[points.astype(int)], color=(0, 0, 0))
    cv2.imshow('roi',img)
    cv2.fillPoly(canny_img, pts=[points.astype(int)], color=(0, 0, 0))
    cv2.fillPoly(mask_img, pts=[points.astype(int)], color=(0, 0, 0))
    cv2.imshow('canny',canny_img)
    cv2.imshow('roi',img)
    cv2.imshow('mask',mask_img)
    mask_img_draw = mask_img.copy()
    img_draw = img.copy()
    cv2.waitKey(0)
    
    print(findStartingPoint(mask_img, canny_img, img_draw,15,50, startY=.95))
    # print(vote)
    # scores = [s[0] for s in vote]
    # idMax = np.argmax(scores)
    # print(vote[idMax])

    # inner_segs, outer_segs = sliding_window(img) #FIXME - stop without median condition 