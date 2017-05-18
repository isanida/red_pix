from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import os
import plotly.graph_objs as go
import plotly.offline as py

from sklearn import preprocessing
from sklearn.manifold import TSNE

import src.preprocessing.image_processing as ip


def img_bgr2rbg(img: np.ndarray) -> np.ndarray:
    """
    Method to convert from bgr to rgb
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def img_to_gray(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# def resize_image(img: np.ndarray, rescaled_dim: Iterable(int, int), interpolation: int=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Method to resize the image
    """
    # img = cv2.resize(img, (rescaled_dim[0], rescaled_dim[1]), interpolation=interpolation)
    # return img


def sample_dataframe(df: pd.DataFrame, size: int=None, fraction: float=None, by: str="type",
                     random_state: int=0) -> pd.DataFrame:

    if size == -1:
        # Return all data
        return df

    df_sample = pd.DataFrame()
    for t in df[by].unique():
        df_sample = df_sample.append(df.loc[df[by] == t, :].sample(n=size, frac=fraction, random_state=random_state),
                                     ignore_index=True)
    if size:
        groups = df_sample.groupby(by=by).size()
        assert len(groups.unique()) == 1, "Samples do not have the same size"
        assert groups.unique().squeeze() == size, "Sample size does not match with the input size"

    return df_sample


def normalize_image(img: np.ndarray, norm_type: int=cv2.NORM_MINMAX):
    if not type(img) == float:
        img = img.astype(float)
    return cv2.normalize(src=img, dst=None, alpha=0.0, beta=1.0, norm_type=norm_type)


def get_unit_vector(vector: np.ndarray):
    return vector / np.linalg.norm(vector)


def flatten_matrix(matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    return matrix.reshape(height, width)


# def transform_image(img: np.ndarray, rescaled_dim: Iterable(int, int), to_gray=False):
#     img = resize_image(img=img, rescaled_dim=rescaled_dim, interpolation=cv2.INTER_LINEAR)
#
#     if to_gray:
#         img = img_to_gray(img=img)
#
#     img_n = normalize_image(img=img, norm_type=cv2.NORM_MINMAX)
#
#     return img_n


def process_images(df: pd.DataFrame, rescaled_dim: Iterable(int, int)=None, depth: int=3) -> Iterable(np.ndarray,
                                                                                                      np.ndarray,
                                                                                                      np.ndarray):
    """
    Function to process all images and store them in one 4 dimensional numpy array, with dimensions
    [n, w, h, d] -> n = number of images, w = width, h =height, d = depth
    :param df: DataFrame containing the file_paths to each image
    :param rescaled_dim: iterable containing the final dimensions of each image
    :param depth: the image depth
    :return: 4D numpy array
    """

    if not rescaled_dim:
        rescaled_dim = [256, 256]

    images_array = np.zeros([len(df), rescaled_dim[0], rescaled_dim[1], depth])
    types = []
    file_names = []

    assert "file_path" in df, "column 'file_path' not in DataFrame"

    for i, f in enumerate(df["file_path"]):
        file_names.append(os.path.split(f)[1])
        types.append(df.loc[df["file_path"] == f, "type"].squeeze())
        img = ip.get_image_data(file_path=f)
        images_array[i, :, :, :] = transform_image(img=img, rescaled_dim=rescaled_dim)

    # imgs_mat = np.array(imgages_array).squeeze()
    # df_img_mat = pd.DataFrame(imgs_mat)
    # df_img_mat.insert(loc=0, column="image", value=file_names)
    # df_img_mat.insert(loc=1, column="type", value=types)

    return images_array, np.array(types), np.array(file_names)


def tsne_embedding(imgs_mat):
    tsne = TSNE(n_components=3, init='random', random_state=101, method='barnes_hut', n_iter=500, verbose=2)
    tsne = tsne.fit_transform(imgs_mat)

    return tsne


def plot_tsne(tsne, types: list, path: str):

    trace1 = go.Scatter3d(x=tsne[:, 0], y=tsne[:, 1], z=tsne[:, 2], mode='markers',
                          marker=dict(sizemode='diameter',
                                      color=preprocessing.LabelEncoder().fit_transform(types),
                                      colorscale='Portland',
                                      colorbar=dict(title='cervix types'),
                                      line=dict(color='rgb(255, 255, 255)'),
                                      opacity=0.9))

    data = [trace1]
    layout = dict(height=800, width=800, title='3D embedding of images')
    fig = dict(data=data, layout=layout)
    py.offline.plot(fig, filename=os.path.join(path, "3D_Bubble.html"))

