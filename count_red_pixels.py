import numpy as np
import cv2
import os
import os.path
from PIL import Image
from matplotlib import pyplot as plt
import glob

path1 = os.path.abspath('../data/train/Type_1')
path2 = os.path.abspath('../data/train/Type_2')
path3 = os.path.abspath('../data/train/Type_3')
folders = [path1, path2, path3]

def load_images():
    for filename in glob.iglob('../data/train/**/*.jpg', recursive=True):
        return filename

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpg']]):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

# all_images = [img for folder in folders for img in load_images_from_folder(folder)]

# for folder in folders:
#    images = load_images_from_folder(folder)

def show_image(img: np.ndarray, img_name: str="image"):
    """
    Method to show the image
    """
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread(os.path.join(path1, '12.jpg'))
print(img.shape)

## count pixels having RGB values in defined range ~ assuming red if RGB value > 200
## Using opencv
# RED_MIN = np.array([200, 200, 200], np.uint8)
# RED_MAX = np.array([255, 255, 255], np.uint8)
#
# dst = cv2.inRange(img, RED_MIN, RED_MAX)
# no_red = cv2.countNonZero(dst)
# print("number of red pixels is: " + str(no_red))
# plt.hist(img.ravel(), 56, [0, 56])
# plt.show()
#
# cv2.namedWindow("opencv", cv2.WINDOW_NORMAL)
# cv2.imshow("opencv", img)
# cv2.waitKey(0)

## Using PIL
# upper = (255, 255, 255)
# lower = (200, 200, 200)
# d = '\n'.join([str(pixel) for pixel in img2.getdata() \
#            if False not in map(operator.lt,lower,pixel) \
#            and False not in map(operator.gt,upper,pixel)])
#
# print(type(d))
# print(d)
# nparr = np.fromstring(d, np.uint8)
# e = cv2.imencode(nparr, cv2.IMREAD_COLOR)

#
# hist = cv2.calcHist([e], [0], None, [56], [0, 56])
# hist, bins = np.histogram(img2.ravel(), 56, [0, 56])
# print(np.sum(hist))
# plt.figure()
# plt.title("histogram")
# plt.xlabel("bins")
# plt.ylabel("number of pixels")
# plt.plot(hist)
# plt.xlim([0, 56])




