import numpy as np
import cv2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import RANSACRegressor, HuberRegressor

def curve_fit(X,y, feature = SplineTransformer(n_knots=4, degree=2), estimator = HuberRegressor()):
    '''
    estimator: [HuberRegressor(), RANSACRegressor(random_state=0)]
    feature: [SplineTransformer(n_knots, degree), 'PolynomialFeatures(degree)', ...]
    '''
    model = make_pipeline(feature, estimator)
    model.fit(X, y)
    return model

def drawopenCV(X,y, X_test, y_pred, viz_edge):
    for i,j in zip(X,y):
        print(i,j)
        cv2.circle(viz_edge, (int(i),int(j)), color=(0,125,0), radius=2)

    for i,j in zip(X_test,y_pred):
        print(i,j)
        cv2.circle(viz_edge, (int(i),int(j)), color=(0,125,125), radius=1)
    return viz_edge

def predict_curve(X,y, model):
    X_test = np.linspace(min(X), max(X), 100)
    y_pred = model.predict(X_test)

    # for i,j in zip(X,y):
    #     print(i,j)
    #     cv2.circle(viz_edge, (int(i),int(j)), color=(0,125,0), radius=2)

    # for i,j in zip(X_test,y_pred):
    #     print(i,j)
    #     cv2.circle(viz_edge, (int(i),int(j)), color=(0,125,125), radius=1)
    return X_test, y_pred

if __name__ == '__main__':
    X,y = np.load('outer.npy')
    h, w = 723,383
    viz_curve = np.zeros((h, w, 3))
    

    # print(cv2.UMat(np.array(list(zip(X,y)), dtype=np.uint8).reshape(-1,1,2)))
    points = cv2.UMat(np.array(list(zip(X,y))).reshape(-1,1,2))
    print(points.get().shape)
    print(type(points.get().shape))
    # ellipse = cv2.fitEllipse(points)
    # model = curve_fit(X, y)
    # X_test = np.linspace(min(X), max(X), 100)
    # y_pred = model.predict(X_test)
    #         # cv2.line(viz_edge, s[:2], s[2:],color=(125,125,125), thickness=2)
    # # viz_edge = np.zeros(img.shape)
    # cv2.imshow(' edge', viz_edge)
    # for i,j in zip(X,y):
    #     print(i,j)
    #     cv2.circle(viz_edge, (int(i),int(j)), color=(0,125,0), radius=2)
    # for i,j in zip(X_test,y_pred):
    #     print(i,j)
    #     cv2.circle(viz_edge, (int(i),int(j)), color=(0,125,125), radius=1)
    # # cv2.ellipse(viz_edge, ellipse, (0, 255, 0), 2)
    X,y = np.load('outer.npy')
    model = curve_fit(X, y)
    X_t, y_t = predict_curve(X, y, model)
    viz_curve = drawopenCV(X, y, X_t, y_t, viz_curve)

    X,y = np.load('inner.npy')
    model = curve_fit(X, y)
    X_t, y_t = predict_curve(X, y, model)
    viz_curve = drawopenCV(X, y, X_t, y_t, viz_curve)
    
    # cv2.imshow('curve pred', )
    cv2.imshow('curve pred', viz_curve)
    # viz_edge = predict_curve(X, y, model)
    # cv2.imshow('predicted edge', viz_edge)

    cv2.waitKey(0)