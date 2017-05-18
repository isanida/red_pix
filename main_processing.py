import src.utils.filemanager as fm
import os


def main():

    path_data = os.path.join(os.path.join(fm.PATH_HOME, "data"))
    path_output = os.path.join(fm.PATH_HOME, "output")
    data_paths = os.listdir(path_data)

    path_img_info = os.path.join(path_output, "img_info.csv")
    path_processed_imgs = os.path.join(path_output, "flattened_imgs.pkl")

    df_img_info = fm.read_csv(path_img_info)
    print(df_img_info.shape)

if __name__ == "__main__":
    main()