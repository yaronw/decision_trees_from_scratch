from .data_processing import *
from .list_helpers import *


DEFAULT_NUM_OF_DISCRETIZING_DIVISIONS = 10  # number of discrete values to use when discretizing a continuous variable

ABALONE_Y_COL_NUM = 8
CAR_ACCEPTABILITY_Y_COL_NUM = 6
SEGMENTATION_Y_COL_NUM = 0

COMPUTER_HARDWARE_Y_COL_NUM = 6
FORESTFILE_Y_COL_NUM = 10
WINE_Y_COL_NUM = 11

def load_abalone_data(path_to_file):
    """ Loads the Abalone data from file. """
    data = pd.read_csv(path_to_file, header=None)

    x_data, y_data = split_x_y(data, ABALONE_Y_COL_NUM)

    x_list = x_data.values.tolist()
    y_list = transpose_matrix(y_data.values.tolist())[0]

    return x_list, y_list


def load_car_evaluation_data(path_to_file):
    """ Loads the Car Acceptability data from file. """
    data = pd.read_csv(path_to_file, header=None)

    x_data, y_data = split_x_y(data, CAR_ACCEPTABILITY_Y_COL_NUM)

    x_list = x_data.values.tolist()
    y_list = transpose_matrix(y_data.values.tolist())[0]

    return x_list, y_list


def load_segmentation_data(path_to_file):
    """ Loads the Image Segmentation data from file. """
    data = pd.read_csv(path_to_file, skiprows=5, header=None)

    x_data, y_data = split_x_y(data, SEGMENTATION_Y_COL_NUM)

    x_list = x_data.values.tolist()
    y_list = transpose_matrix(y_data.values.tolist())[0]

    return x_list, y_list


def load_computer_hardware_data(path_to_file):
    """ Loads the Computer Hardware data from file. """
    data = pd.read_csv(path_to_file, header=None)
    data = data.drop(columns=[0, 1, 9])  # remove non-predictive vendor and models, as well estimated performance value from paper

    x_data, y_data = split_x_y(data, COMPUTER_HARDWARE_Y_COL_NUM)

    x_list = x_data.values.tolist()
    y_list = transpose_matrix(y_data.values.tolist())[0]

    return x_list, y_list


def load_forestfires_data(path_to_file):
    """ Loads the Forest Fires data from file. """
    data = pd.read_csv(path_to_file)
    data = data.drop(columns=["month", "day"])  # remove date data

    x_data, y_data = split_x_y(data, FORESTFILE_Y_COL_NUM)

    x_list = x_data.values.tolist()
    y_list = transpose_matrix(y_data.values.tolist())[0]

    return x_list, y_list


def load_wine_quality_data(path_to_file):
    """ Loads the Wine data from file. """
    data = pd.read_csv(path_to_file, sep=';')

    x_data, y_data = split_x_y(data, WINE_Y_COL_NUM)

    x_list = x_data.values.tolist()
    y_list = transpose_matrix(y_data.values.tolist())[0]

    return x_list, y_list
