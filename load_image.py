# import cv2
# from matplotlib import pyplot as plt
#
# path = 'C:/Users/Ioanna/Nextcloud/hpp_healthcare_demo/Kaggle_Competition/data/train/Type_3/23.jpg'
# image = cv2.imread(path)
# h, w, d = image.shape
# print(h, w, d)
#
# x = 3000
# y = 1000
# d = 900
#
# i = image[x:x+d, y:y+d]
# #plt.imshow(i)
# #cv2.waitKey(0)
#
# hist = cv2.calcHist([i], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()
#
# print(hist)

import cv2
import os
import glob
import numpy as np
from skimage import io
import pandas as pd

# root = "C:/Users/Ioanna/Desktop/kaggle/data/train"
root = "C:/Users/Ioanna/Desktop/kaggle/data/step1/ori"
folders = ["Type_1", "Type_2", "Type_3"]
extension = "*.jpg"



def file_is_valid(filename):
    try:
        io.imread(filename)
        return True
    except:
        return False

def compute_red_histogram(root, folders, extension):
    X = [] #2D array : rows=number of images
    y = [] #1D array : numeric class labels (0 for Type_1, 1 for Type_2, 2 for Type_3)
    for n, imtype in enumerate(folders):
        filenames = glob.glob(os.path.join(root, imtype, extension))
        for fn in filter(file_is_valid, filenames):
            print(fn)
            image = io.imread(fn)
            img = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            red = img[:, :, 0]
            h, _ = np.histogram(red, bins=np.arange(257), normed=True)
            X.append(h)
            y.append(n)
    return np.vstack(X), np.array(y)

X, y = compute_red_histogram(root, folders, extension)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# train svm classifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(SVC(probability=True))
clf.fit(X_train, y_train)

# y_test
# clf.predict(X_test)
# y_test == clf.predict(X_test)
score = clf.predict_proba(X_test)

# clf.predict_proba(X_test)
prediction = pd.DataFrame([y_test, score])
prediction.to_csv('prediction.csv')

# np.savetxt(root, prediction, score, delimiter=',')