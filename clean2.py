import cv2
import imageio
import numpy as np
from curve_fitting import curve_fit, predict_curve, drawopenCV
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, FunctionTransformer
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from rdp import rdp


l_r_ratio = 2.0449
def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel
if __name__ == '__main__':
    segnet_res_path = '/home/pronton/SeAH/output/mask_combine_cam4and3_20230907_143145_mod10/'
    left_cam_path = '/home/pronton/SeAH/data/cam3_20230907_143145_mod10/'
    right_cam_path = '/home/pronton/SeAH/data/cam4_20230907_143145_mod10/'
    # impath = '/home/pronton/SeAH/data/cam3_20230907_143145/'
    # img_paths = [segnet_res_path + 'combine' + str(idx) + '.jpg' for idx in list(range(0,1020,10)[::-1])]
    scale = .3
    image_lst = []
    for idx_img in list(range(0,1020,10)):
        path = segnet_res_path + 'combine' + str(idx_img) + '.jpg' 
        outputMask = cv2.imread(path)
        # outputMask = cv2.resize(outputMask, (0,0), fx=scale, fy=scale)
        h,w = outputMask.shape[:2]
        print('shape', h, w)
        # leftImg = outputMask[:, :w//2]
        # rightImg = outputMask
        # blend_img = cv2.addWeighted(leftImg, 0.8, rightImg, 0.2, 1.0)
        rightImg = cv2.cvtColor(outputMask, cv2.COLOR_BGR2GRAY)
        rightImg = cv2.resize(rightImg, (0,0), fx=scale, fy=scale)
        # cv2.waitKey(0)
        # continue


        rightImg = cv2.blur(rightImg, (10,10))
        cv2.imshow('blur', rightImg)
        # (th, outputMask) = cv2.threshold(outputMask, 245, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        opening = cv2.morphologyEx(rightImg, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel=(3,3), iterations=1)
        # gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel=(5,5), iterations=1)
        # skeleton = cv2.morphologyEx(opening, cv2.MORPH_S, kernel=(5,5), iterations=1)
        (th, opening) = cv2.threshold(opening, 245, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        skeleton = skeletonize(opening)
        (th, skeleton) = cv2.threshold(skeleton, 245, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('opening', opening)
        # cv2.imshow('opening+closing', closing)
        # cv2.imshow('gradient', gradient)
        cv2.imshow('skel', skeleton)

        contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        lenContours = np.asarray([len(c) for c in contours])
        # sortidx = np.argsort(lenContours)[::-1]
        sortidx = np.where(lenContours > 40*scale)[0]
        # print(sortidx)
        # print(len(contours[sortidx[0]]))

        contour_img = np.zeros_like(skeleton)
        cv2.drawContours(contour_img, contours, -1, 255, 3)
        cv2.imshow('all contour', contour_img)
        contour_img = np.zeros_like(skeleton)
        for i in sortidx:
            cv2.drawContours(contour_img, contours[i], -1, 255, 3)
        cv2.imshow('filter contour', contour_img)
        # cv2.waitKey(0)
        skeleton = skeletonize(contour_img)
        (th, skeleton) = cv2.threshold(skeleton, 245, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('skel after contour', contour_img)

        points = []
        count = 0
        h_sub,w_sub = skeleton.shape[:2]
        for y in range(h_sub):
            for x in range(w_sub):
                if skeleton[y][x]==255:
                    points.append([x,y])
        points = np.asarray(points)
        # viz_curve = np.zeros((h_sub, w_sub, 3))
        # print('prev len', len(points))
        # X= points[:, 0].reshape(-1, 1)
        # y= points[:, 1].reshape(-1, 1)
        # viz_curve = drawopenCV(X, y, [], [],  viz_curve)
        # cv2.imshow('before rdp', viz_curve)
        # viz_curve = np.zeros((h_sub, w_sub, 3))

        # points = rdp(points, epsilon=0.1)
        # print('after len', len(points))
        # X= points[:, 0].reshape(-1, 1)
        # y= points[:, 1].reshape(-1, 1)
        # viz_curve = drawopenCV(X, y, [], [],  viz_curve)
        # cv2.imshow('after rdp', viz_curve)

        if len(points)< 1:
            continue

        X= points[:, 0].reshape(-1, 1)
        y= points[:, 1].reshape(-1, 1)
        # right
        idx = np.where(X>int(w_sub/l_r_ratio)) # FIXME: wrong idx w_left : w_right = 883 : 845
        X_right = X[idx].reshape(-1,1)
        y_right = y[idx].reshape(-1,1)
        X_right,y_right = y_right,X_right

        viz_curve = np.zeros((h_sub, w_sub, 3))
        try:
            X_t, y_t = predict_curve(X_right, y_right, feature=SplineTransformer(n_knots=2, degree=2), estimator=HuberRegressor())
            # y_top_idx = np.argmin(X_t) # highest
            # X_top = X_t[y_top_idx]
            # y_top = y_t[y_top_idx]
            # print(X_top, y_top)
            contact_point_X = w_sub/l_r_ratio + 70 / 845 * w_sub/l_r_ratio
            diff = y_t[0] - contact_point_X 
            cv2.circle(viz_curve, (int(y_t[0]), int(X_t[0])), radius=2, color=(255,0,255), thickness=-1)
            # cv2.line(viz_curve, (int(y_t[0]), int(X_t[0])), (int(contact_point_X), int(X_t[0])), color=(255,0,255), thickness=2)
            cv2.line(viz_curve, (int(y_t[0]), int(X_t[0])), (int(contact_point_X), int(X_t[0])), color=(255,255,255), thickness=2)
            cv2.putText(viz_curve, f'{np.round(diff,2)}', (int(contact_point_X), int(X_t[0])+20),cv2.FONT_HERSHEY_SIMPLEX, .3, (255,255,255,), 1, cv2.LINE_AA)
            viz_curve = drawopenCV(y_right, X_right, y_t, X_t,  viz_curve)
            # y_top_idx = np.argmin(y_right) # highest
            # X_top = X_right[y_top_idx]
            # y_top = y_right[y_top_idx]
        except:
            pass
        
        # left
        idx = np.where(X<=int(w_sub/l_r_ratio))
        X_left = X[idx].reshape(-1,1)
        y_left = y[idx].reshape(-1,1)
        X_left,y_left = y_left,X_left

        # viz_curve = np.zeros((h, w, 3))
        try:
            X_t, y_t = predict_curve(X_left, y_left, feature=SplineTransformer(n_knots=2, degree=3), estimator=HuberRegressor())
            contact_point_X = 813 / 883 * w_sub/l_r_ratio
            diff = contact_point_X - y_t[0]
            print(int(contact_point_X))
            # cv2.circle(viz_curve, (int(y_t[0]), int(X_t[0])), radius=2, color=(255,0,255), thickness=-1)
            cv2.line(viz_curve, (int(y_t[0]), int(X_t[0])), (int(contact_point_X), int(X_t[0])), color=(255,255,255), thickness=2)
            viz_curve = drawopenCV(y_left, X_left, y_t, X_t,  viz_curve)
            cv2.putText(viz_curve, f'{np.round(diff,2)}', (int(contact_point_X), int(X_t[0])+20),cv2.FONT_HERSHEY_SIMPLEX, .3, (255,255,255,), 1, cv2.LINE_AA)
            viz_curve = drawopenCV(y_right, X_right, y_t, X_t,  viz_curve)
        except:
            pass
        cv2.imshow('viz', viz_curve)


        # #down
        # idx = np.where(y>h//2)
        # X_lower = X[idx].reshape(-1,1)
        # y_lower = y[idx].reshape(-1,1)
        # # X,y = y,X

        # # viz_curve = np.zeros((h,w, 3))
        # X_t, y_t = predict_curve(X_lower, y_lower)
        # viz_curve = drawopenCV(X_lower,y_lower, X_t, y_t, viz_curve)
        # cv2.imshow('viz', viz_curve)
        
        # cv2.waitKey(0)
        dim = (w,h)
        viz_curve = cv2.resize(viz_curve, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow('viz_curve', viz_curve)
        # cv2.waitKey(0)
        viz_curve=  viz_curve.astype(np.uint8)
        print(dim)
        cv2.imshow('viz_curve', viz_curve)
        leftImg = np.ones((h,w,3), dtype=np.uint8)

        left_cam = cv2.imread(f'{left_cam_path}{idx_img}.jpg')
        left_cam = cv2.resize(left_cam, (w//2,h), interpolation=cv2.INTER_AREA)
        right_cam = cv2.imread(f'{right_cam_path}{idx_img}.jpg')
        
        leftImg[:, :845] = left_cam
        leftImg[:, 845:] = right_cam


        cv2.imshow('comb', leftImg)
        blend_img = cv2.addWeighted(leftImg, 1, viz_curve, 1.0, 1.0)
        # cv2.imshow('blend_viz', blend_img)
        # cv2.waitKey(0)

        image_lst.append(cv2.cvtColor(blend_img, cv2.COLOR_BGR2RGB))
    imageio.mimsave('demo_segnet_6fps_distance_estimation.mp4', image_lst, fps=6)