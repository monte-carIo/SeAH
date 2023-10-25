import numpy as np
import cv2

X = np.load('X_out.npy')
y = np.load('y_out.npy').reshape(-1,1)
print(X.shape, y.shape)
# with open('X.npy', 'wb') as f:
#     np.save(f, X)
# with open('y.npy', 'wb') as f:
#     np.save(f, y)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import RANSACRegressor, HuberRegressor

# estimator = HuberRegressor()
estimator = RANSACRegressor(random_state=0)
# model = make_pipeline(PolynomialFeatures(2), estimator)
model = make_pipeline(SplineTransformer(n_knots=5, degree=1), estimator)
# print(cv2.UMat(np.array(list(zip(X,y)), dtype=np.uint8).reshape(-1,1,2)))
points = cv2.UMat(np.array(list(zip(X,y))).reshape(-1,1,2))
print(points.get().shape)
print(type(points.get().shape))
ellipse = cv2.fitEllipse(points)

model.fit(X, y)
y_pred = model.predict(X)

viz_edge = np.zeros((788, 908, 3))
        # cv2.line(viz_edge, s[:2], s[2:],color=(125,125,125), thickness=2)
# viz_edge = np.zeros(img.shape)
cv2.imshow(' edge', viz_edge)
for i,j,z in zip(X,y_pred,y):
    print(i,j)
    cv2.circle(viz_edge, (int(i),int(j)), color=(0,125,125), radius=2)
    cv2.circle(viz_edge, (int(i),int(z)), color=(125,125,125), radius=2)
cv2.ellipse(viz_edge, ellipse, (0, 255, 0), 2)
cv2.imshow('predicted edge', viz_edge)

cv2.waitKey(5000)