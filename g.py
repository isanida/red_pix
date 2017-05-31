import numpy as np
import cv2
import os.path
import glob
import matplotlib.pyplot as plt

histogram = {}

#output dic
out = {
    1: {},
    2: {},
    3: {},
}
test_im = "C:/Users/Ioanna/Desktop/kaggle/data/test/test.jpg"

for t in [1]:

    #load_files
    files = glob.glob(os.path.join("..", "data", "train", "Type_{}".format(t), "*.jpg"))
    no_files = len(files)

    #iterate and read
    for n, file in enumerate(files):
        try:
            image = cv2.imread(file)
            img = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #feature : count red distribution
            red_min = np.array([80, 100, 100])
            red_max = np.array([179, 255, 255])
            mask = cv2.inRange(hsv, red_min, red_max)
            res = cv2.bitwise_and(img, img, mask=mask)
            ratio_red = cv2.countNonZero(mask)/(img.size/3)
            s = np.round(ratio_red*100, 2)
            print("red pixel percentage: ", s)
            cv2.imshow("images", np.hstack([img, res]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # features : histograms
            plt.hist(mask.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.legend('histogram', loc='upper left')

            # histograms[file] = hist
            plt.show()

            #save key
            out[t][file] = histogram

            # dist = cv2.compareHist(hist_test, hist, cv2.HISTCMP_CORREL)
            # print(dist)


            print(file, t, "-files left", no_files - n)


        except Exception as e:
            print(e)
            print(file)


