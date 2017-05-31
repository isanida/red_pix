import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2

root = "C:/Users/Ioanna/Desktop/kaggle/data/additional_data"  # Change this appropriately
folders = ['Type_1', 'Type_2', 'Type_3']
extension = '*.jpg'  # Change if necessary
threshold = 15 # Adjust to fit your needs

n_bins = 5  # Tune these values to customize the plot
width = 2.
colors = ['cyan', 'magenta', 'yellow']
edges = np.linspace(0, 100, n_bins+1)
centers = .5*(edges[:-1]+ edges[1:])

# This is just a convenience class used to encapsulate data
class img_type(object):
    def __init__(self, folder, color):
        self.folder = folder
        self.percents = []
        self.color = color

lst = [img_type(f, c) for f, c in zip(folders, colors)]

fig, ax = plt.subplots()

for n, obj in enumerate(lst):
    filenames = glob.glob(os.path.join(root, obj.folder, extension))

    for fn in filenames:
        img = cv2.imread(fn)
        red = img[:, :, 2]
        obj.percents.append(100.*np.sum(red >= threshold)/red.size)

    h, _ = np.histogram(obj.percents, bins=edges)
    h = np.float64(h)
    h /= h.sum()
    h *= 100.
    ax.bar(centers + (n - .5*len(lst))*width, h, width, color=obj.color)

ax.legend(folders)
ax.set_xlabel('% of pixels whose red component is >= threshold')
ax.set_ylabel('% of images')
plt.show()