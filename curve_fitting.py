import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, FunctionTransformer
from sklearn.linear_model import RANSACRegressor, HuberRegressor

from ellip import ellip_model, ellip_predict
from bezier import get_bezier_parameters, bezier_curve

def curve_fit(X,y, feature = SplineTransformer(n_knots=3, degree=3), estimator = HuberRegressor()):
    '''
    estimator: [HuberRegressor(), RANSACRegressor(random_state=0)]
    feature: [SplineTransformer(n_knots, degree), 'PolynomialFeatures(degree)', ...]
    '''
    model = make_pipeline(feature, estimator)
    model.fit(X, y)
    return model


def drawopenCV(X,y, X_test, y_pred, viz_edge, dataColor = (255,255,255), predictColor = (0,125,0)):
    for i,j in zip(X,y):
        # print(i,j)
        cv2.circle(viz_edge, (int(i),int(j)), color=dataColor, radius=2)

    for i,j in zip(X_test,y_pred):
        # print(i,j)
        cv2.circle(viz_edge, (int(i),int(j)), color=predictColor, radius=1)
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

def ellipse_cv(X, y, viz_curve = np.zeros((723, 383, 3))):
    points = cv2.UMat(np.array(list(zip(X,y))).reshape(-1,1,2))
    ellipse = cv2.fitEllipse(points)
    for i,j in zip(X,y):
        # print(i,j)
        cv2.circle(viz_curve, (int(i),int(j)), color=(255,255,255), radius=2)
    # for i,j in zip(X,y):
    #     # print(i,j)
    #     cv2.circle(viz_curve, (int(i),int(j)), color=(0,125,125), radius=1)
    # (x, y), (MA, ma), angle  = ellipse
    # cv2.ellipse(viz_curve, center=[int(x),int(y)], axes=(int(MA/2), int(ma/2)), angle=angle, startAngle=220, endAngle=360, color=(125, 255, 125), thickness=2)
    cv2.ellipse(viz_curve, ellipse, color=(0, 255, 0), thickness=2)
    cv2.imshow('ellips pred', viz_curve)
    return ellipse

if __name__ == '__main__':
    y,X = np.load('outer.npy')
    h, w = 723,383
    viz_curve = np.zeros((h, w, 3))
    
    print(ellipse_cv(y,X))
    viz_curve = np.zeros((h, w, 3))
    y,X = np.load('outer.npy')
    model = curve_fit(X, y, SplineTransformer(n_knots=3, degree=3))
    X_t, y_t = predict_curve(X, y, model)
    viz_curve = drawopenCV(y, X, y_t, X_t,  viz_curve)

    y,X = np.load('inner.npy')
    model = curve_fit(X, y, SplineTransformer(n_knots=3, degree=3))
    X_t, y_t = predict_curve(X, y, model)
    viz_curve = drawopenCV(y, X, y_t, X_t, viz_curve, predictColor=(0,0,125))
    
    # cv2.imshow('curve pred', )
    cv2.imshow('curve pred', viz_curve)
    # viz_edge = predict_curve(X, y, model)
    # cv2.imshow('predicted edge', viz_edge)
    cv2.waitKey(0)

    X,y = np.load('outer.npy')
    # X = X.reshape(-1)
    # y = y.reshape(-1)
    # data = get_bezier_parameters(X.reshape(-1), y.reshape(-1), degree=4)
    # x_val = [x[0] for x in data]
    # y_val = [x[1] for x in data]
    # # print(data)
    # # Plot the control points
    # plt.scatter(X.astype(int),y.astype(int), s=1, label='Control Points')
    # plt.plot(x_val,y_val,'k--o', label='Control Points')
    # # Plot the resulting Bezier curve
    # xvals, yvals = bezier_curve(data, nTimes=1000)
    # plt.plot(xvals, yvals, 'b-', label='B Curve')
    # plt.xlim(0,w)
    # plt.ylim(0,h)
    # plt.gca().invert_yaxis()
    # plt.legend()

    x = ellip_model(y,X)
    Z_coord = ellip_predict(y,X,x)
    plt.show()
