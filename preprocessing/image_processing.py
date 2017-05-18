import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns


def get_size_info(img: np.ndarray) -> pd.Series:
    return pd.Series({"height": img.shape[0],
                      "width": img.shape[1],
                      "depth": img.shape[2]})


def get_train_image_info(train_path: str, df_current: pd.DataFrame) -> pd.DataFrame:
    train_dirs = os.listdir(train_path)
    train_dirs = [os.path.join(train_path, d) for d in train_dirs if os.path.isdir(os.path.join(train_path, d))]

    all_images = []
    for path in train_dirs:
        all_images.extend([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                          and not f.endswith("DS_Store")])
    df = pd.DataFrame({"file_path": all_images})

    if df_current.shape[0] == df.shape[0]:
        print("The currently stored DataFrame already has the info of all available images")
        return df_current

    df["file_extension"] = df.apply(lambda row: row["file_path"].split(".")[-1], axis=1)
    df["type"] = df.apply(lambda row: row["file_path"].split(os.sep)[-2], axis=1)
    df[["depth", "height", "width"]] = df["file_path"].apply(lambda row: get_size_info(img_path=row))

    return df







