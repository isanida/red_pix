import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def show_image(img: np.ndarray, img_name: str="image"):
    """
    Method to show the image
    """
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_image_info(df: pd.DataFrame):

    assert all(x in df.columns for x in ["type", "file_extension", "count", "file_path"])

    type_aggregation = df.groupby(['type', 'file_extension']).agg('count').sort_values(by="file_path")
    type_aggregation_p = type_aggregation.apply(lambda row: 1.0 * row['file_path'] / df.shape[0], axis=1)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    type_aggregation.plot.barh(ax=axes[0])
    axes[0].set_xlabel("image count")
    type_aggregation_p.plot.barh(ax=axes[1])
    axes[1].set_xlabel("training size fraction")
    plt.show()

    shapes_df_grouped = df.groupby(by=['depth', 'width', 'height', 'type']).size().to_frame('group_size').\
        reset_index().sort_values(['type', 'group_size'], ascending=False)

    shapes_df_grouped['size_with_type'] = shapes_df_grouped.apply(
        lambda row: '{}-{}-{}'.format(row["width"], row["height"], row["type"]), axis=1)
    shapes_df_grouped = shapes_df_grouped.set_index(shapes_df_grouped['size_with_type'].values)
    shapes_df_grouped['count'] = shapes_df_grouped["group_size"]

    plt.figure(figsize=(10, 8))
    # shapes_df_grouped['count'].plot.barh(figsize=(10,8))
    sns.barplot(x="count", y="size_with_type", data=shapes_df_grouped)