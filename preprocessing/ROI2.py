import cv2

# ROI v0.1
# vid1 = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam3_20230907_143145.avi') 
# vid1 = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/13/35/cam3_20230907_133530.avi') 
# vid1 = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/13/35/cam4_20230907_133530.avi') 
vid1 = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam3_20230907_143145.avi') 
vid1 = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam4_20230907_143145.avi') 
ret, frame = vid1.read() 
# print(vid1.get(cv2.CAP_PROP_FPS))
# 6fps => 40 frames -> 7 seconds
# print(frame.shape)
# exit()
h,w = frame.shape[:2]
scale = 1
w = w * scale
h = h * scale
i = 0
# sec =0
# sec = 11 # cam4 is 6 seconds later of cam 3
# cv2.CAP_PROP_POS_FRAMES
# vid1.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
# vid1.set(cv2.CAP_PROP_POS_FRAMES,12)
# vid1.set(cv2.CAP_PROP_POS_FRAMES,12)
vid1.set(cv2.CAP_PROP_POS_FRAMES,30)
while(vid1.isOpened()):
    ret, frame = vid1.read() 
    if ret:
        frame = cv2.resize(frame, (0, 0), fx = scale, fy = scale) 
        shift_y = 0.026
        shiftx_cam4 = -0.01
        shiftx_cam3 = 0.03
        constantY = .670
        shiftx2_cam4 = 0.01

        # cam 4 (right)
        y1, y2 = int(h*(.08-shift_y)), int(h*(.08-shift_y)) + int(h*constantY)
        x1, x2 = int(w*(.25+shiftx_cam4)), int(w*(.45+shiftx2_cam4))
        ROI = frame[y1:y2, x1:x2]

        # cv2.imshow('cam right',ROI)

        # cam 3 (left)
        #y1, y2 = int(h*.08), int(h*.08) + int(h*constantY)
        #x1, x2 = int(w*.5), int(w*(.7+shiftx_cam3))
        #ROI = frame[y1:y2, x1:x2]

        if i % 10 == 0:
            cv2.imwrite(f'data/cam4_20230907_143145_mod10/{i}.jpg', ROI)
        # cv2.imshow('roi', ROI)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        i+=1
    else:
        break
vid1.release() 
cv2.destroyAllWindows() 
