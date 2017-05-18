import os
import gzip

import cv2
import pandas as pd
import pickle
import numpy as np


PATH_HOME = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SEP = ";"
DTYPE_DEFAULT = {'file_path': str, 'file_extension': str, 'type': str, 'height': int, 'width': int,
                 'depth': int}


def create_dir(*args):
    folder_path = os.path.join(*args)
    if os.path.exists(folder_path) and os.path.isfile(folder_path):
        print("Warning: {} was a file (replaced it with a folder)".format(folder_path))
        os.remove(folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def get_file_path(folder_path, filename):
    return os.path.join(folder_path, filename)


def add_file_extension(path: str, extension: str=".csv") -> str:

    if path[-4:] != extension and path[-4:].startswith("."):
        path = path.replace(path[-4:], extension)
    elif path[-4:] != extension:
        path += extension

    return path


# Reading and writing files

def write_csv(df: pd.DataFrame, path: str, sep=SEP, header=True, mode="w"):
    """
    Write pandas dataframe to csv.

    Keyword arguments:
    path -- Folder path if name is not None,
            File path if name is None
    name -- Path to file inside folderpath.
            If none, then path should speficy the file
    round_floats -- round the columns containing floats to 3 decimals
    """

    path = add_file_extension(path=path, extension=".csv")

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Warning: could not write empty df to {}".format(path))
        return

    if os.path.dirname(path) != '' and not os.path.isdir(os.path.dirname(path)):
        create_dir(os.path.dirname(path))

    for k, v in DTYPE_DEFAULT.items():
        if k in df.columns:
            if not df[k].isnull().any():
                df[k] = df[k].astype(v)

    print("Writing csv to {}".format(os.sep.join(os.path.normpath(path).split(os.sep)[-3:])))
    df.to_csv(path, sep=sep, index=False, float_format='%.3f', header=header, mode=mode)


def write_pickle(data: np.ndarray, path: str, compress: bool=False):
    """
    Pickle pandas DataFrame
    """
    path = add_file_extension(path=path, extension=".pkl")

    if os.path.dirname(path) != '' and not os.path.isdir(os.path.dirname(path)):
        create_dir(os.path.dirname(path))

    if compress:
        path = add_file_extension(path, extension=".gz")
        print("Writing compressed pickle to {}".format(os.sep.join(os.path.normpath(path).split(os.sep)[-3:])))
        f = gzip.open(path, 'wb')
        pickle.dump(data, f)
    else:
        print("Writing pickle to {}".format(os.sep.join(os.path.normpath(path).split(os.sep)[-3:])))
        pickle.dump(data)


def read_csv(file_path: str, sep: str=SEP, verbose=True, dtype={}, **kwargs):
    """
    Read file and return as pandas dataframe

    Default dtypes for known columns are specified in this function, for optimization and data consistency,
    but can be overridden using the dtype argument.
    """
    if not os.path.isfile(file_path):
        return pd.DataFrame([])

    if file_path is not None:
        if file_path[-4:] != '.csv':
            file_path += '.csv'

    for k, v in DTYPE_DEFAULT.items():
        dtype[k] = dtype[k] if k in dtype else v
    if verbose:
        print('Read csv:', file_path)

    df = pd.read_csv(file_path, sep=sep, dtype=dtype, **kwargs)

    return df


def read_pickle(path: str, compressed: bool=False):
    """
    Read file and return as pandas dataframe

    Default dtypes for known columns are specified in this function, for optimization and data consistency,
    but can be overridden using the dtype argument.
    """

    if compressed:
        path = add_file_extension(path=path, extension=".gz")
        if not os.path.isfile(path):
            return pd.DataFrame([])
        with gzip.open(path, "rb") as f:
            df = pickle.load(f)
        return df

    path = add_file_extension(path=path, extension=".pkl")
    if not os.path.isfile(path):
        return pd.DataFrame([])

    return pd.read_pickle(path)
