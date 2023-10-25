import cv2

vid = cv2.VideoCapture('/media/pronton/D/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam3_20230907_143145.avi') 
vid = cv2.VideoCapture('/media/pronton/D/SeAH Steel Vision System-20230929T072146Z-001/SeAH Steel Vision System/data/14-002/31/cam4_20230907_143145.avi') 
ret, frame = vid.read() 
print(frame.shape)
h,w = frame.shape[:2]
scale = 1
w = w * scale
h = h * scale
i = 0
while(True):
    ret, frame = vid.read() 
    frame = cv2.resize(frame, (0, 0), fx = scale, fy = scale) 
    # cam4
    y1, y2 = int(h*.08), int(h*.75)
    x1, x2 = int(w*.25), int(w*.45)

    # cam3
    # y1, y2 = int(h*.08), int(h*.75)
    # x1, x2 = int(w*.5), int(w*.7)
    ROI = frame[y1:y2, x1:x2]

    if i %10 == 0:
        cv2.imwrite(f'cam4_20230907_143145/{i}.jpg', ROI)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    i+=1
vid.release() 
cv2.destroyAllWindows() 