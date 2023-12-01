import sys
import cv2

def getFrame(frame, path):
    vidcap = cv2.VideoCapture(path) 
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    hasFrames,image = vidcap.read()
    if hasFrames:
        return image
    return hasFrames
# getFrame(10)
def ROI(left, right):
    h,w =  left.shape[:2]

    # cam 4 (right)
    shift_y = 0.026
    shiftx_cam4 = -0.01
    shiftx_cam3 = 0.03
    shiftx2_cam4 = 0.01
    constantY = .670
    # shiftx = float(sys.argv[2])
    # y1, y2 = int(h*(.08-shift_y)), int(h*(.75-shift_y))
    y1, y2 = int(h*(.08-shift_y)), int(h*(.08-shift_y)) + int(h*constantY)
    x1, x2 = int(w*(.25+shiftx_cam4)), int(w*(.45+shiftx2_cam4))
    ROI_r = right[y1:y2, x1:x2]
    # cv2.imshow('cam right',ROI)

    # cam 3 (left)
    y1, y2 = int(h*.08), int(h*.08) + int(h*constantY)
    x1, x2 = int(w*.5), int(w*(.7+shiftx_cam3))
    ROI_l = left[y1:y2, x1:x2]
    return ROI_l, ROI_r

if __name__ == '__main__':
    import numpy as np
    # vidcap = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam3_20230907_143145.avi') 
    # vidcap = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam4_20230907_143145.avi') 
    # processed_fps = vidcap.get(cv2.CAP_PROP_FPS)
    leftimg = getFrame(700,'/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam3_20230907_143145.avi')
    rightimg = getFrame(780,'/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam4_20230907_143145.avi')
    roil, roir = ROI(leftimg, rightimg)
    
    # cv2.imshow('left_img', leftimg)
    # cv2.imshow('right_img', rightimg)
    # imgleft = cv2.imread('130_3.jpg')
    # imgright = cv2.imread('135_4.jpg')
    # roil, roir = ROI(imgleft, imgright)
    # roir = cv2.imread('data/cam4_20230907_143145/290.jpg')
    # roil = cv2.imread('data/cam3_20230907_143145/290.jpg')
    h_l,w_l = roil.shape[:2]
    h_r,w_r = roir.shape[:2]
    print(h_l, h_r)
    assert h_l == h_r, 'Error: Different height'

    combined = np.zeros((h_l, w_l+w_r,3), dtype = np.uint8)
    combined[:h_l, :w_l] = roil
    combined[:h_l, w_l: w_l+w_r] = roir
    cv2.imshow('combined', combined)
    cv2.waitKey(0)