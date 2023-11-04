
import cv2

# cap = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam3_20230907_143145.avi')

cap = cv2.VideoCapture('/media/pronton/C/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam4_20230907_143145.avi')
mog = cv2.createBackgroundSubtractorMOG2()
mog = cv2.createBackgroundSubtractorKNN()

scale = .3
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx = scale, fy = scale)

    h,w = frame.shape[:2]
    y1, y2 = int(h*.08), int(h*.75)
    x1, x2 = int(w*.25), int(w*.45)
    frame = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fgmask = mog.apply(gray)
    
    # Apply morphological operations to reduce noise and fill gaps
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    
    cv2.imshow('Foreground Mask', fgmask)
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()