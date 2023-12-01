import cv2

if __name__ == '__main__':
    impath = '/home/pronton/SeAH/data/cam4_20230907_133530/652.jpg'
    # impath = '/home/pronton/SeAH/data/cam3_20230907_133530/596.jpg'
    # impath = '/home/pronton/SeAH/data/cam3_20230907_143145/'
    # img_paths = [impath + 'combine' + str(idx) + '.jpg' for idx in list(range(1,650,2)[::-1])]
    # scale = .3
    outputMask = cv2.imread(impath)
    h,w = outputMask.shape[:2]
    print(w,h)
    cv2.imshow('s', outputMask)
    # x = 70 / 845
    # x = 813 / 883
    cv2.waitKey(0)