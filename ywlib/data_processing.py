import pandas as pd
import statistics as stat
from .list_helpers import *


def discretize_value(value, min, max, num_of_divisions):
    """ Discretizes a continuous (floating point) value by dividing its range into equal intervals and
    returning the number of the interval where the value falls as an integer.

    :param value:  value to discretize
    :param min:  lower bound of the range, i.e. the minimum possible number the value can take
    :param max:  upper bound of the range, i.e. the maximum possible number the value can take
    :param num_of_divisions:  the number of intervals to discretize into
    :return:  the number of the interval (from 0 to num_of_divisions - 1) where the value falls.
    """
    interval = max - min
    if interval == 0:
        return 0  # edge case - only one number possible

    division_size = interval/num_of_divisions
    division = math.floor((value - min)/division_size)

    return division


def discretize_column(data_frame, column_number, num_of_divisions):
    """  Discretizes a whole column of a data frame.  Uses the minimum and maximum numbers in the column as the
    possible range of values.

    :param data_frame:  data frame on which to operate
    :param column_number:  column number in the data frame to discretize
    :param num_of_divisions:  number of divisions to discretize into
    :return:  N/A  (alters the values in data_frame)
    """
    col = data_frame.iloc[:, [column_number]]
    col_min = col.min().iloc[0]
    col_max = col.max().iloc[0]
    new_col = col.applymap(lambda val: discretize_value(val, col_min, col_max, num_of_divisions))
    data_frame.update(new_col)


def discretize_columns(data_frame, column_numbers, num_of_divisions):
    """ Discretizes multiple columns in a data frame.

    :param data_frame:  data frame on which to operate
    :param column_numbers:  a list of column numbers to discretize
    :param num_of_divisions:  number of divisions to discretize into
    :return:  N/A  (alters the values in data_frame)
    """
    for col_num in column_numbers:
        discretize_column(data_frame, col_num, num_of_divisions)


def replace_missing_with_col_mode(data_frame, missing_value):
    """ Replaces missing values in each column of a data frame with the respective mode value of the column.

    :param data_frame:  data frame on which to operate
    :param missing_value:  value that marks a missing value
    :return:  N/A  (alters the values in data_frame)
    """
    for col_index in data_frame:
        col = data_frame[col_index]
        new_col = col.apply(lambda val: col.mode()[0] if val == missing_value else val)
        data_frame[col_index].update(new_col)


def one_hot_encode(value, domain):
    """ Returns the one-hot code of a discrete domain of values.
    If the domain has two values, it returns a single-value binary code.

    For example, if 'value' is 'b' and 'domain' is ['a', 'b', 'c'], then [0, 1, 0] is returned.
    If 'value' is 'b' and 'domain' is ['a', 'b'], then [1] is returned.

    :param value:  value to encode
    :param domain:  domain of possible values
    :return:  one-hot code
    """
    if len(domain) < 3:  # Domains that have two elements, are coded as a single binary value.
        code = [domain.index(value)]
    else:  # Otherwise, they are coded as a list of zeroes with a single 1 placed at the index to the domain list.
        code = [0]*len(domain)
        code[domain.index(value)] = 1

    return code


def one_hot_decode(domain, code):
    """ Performs the inverse operation of one_hot_encode(), i.e. converts a one-hot code to the value it represents.

    For example, if 'class_domain' is ['a', 'b', 'c'] and 'code' is [0, 1, 0], this function returns 'b'.

    :param domain:  domain of possible values
    :param code:  the code to decode
    :return:  value the code represents
    """
    if len(code) == 1:  # If the code is a list of one element, then it is the index to the domain of classes.
        the_class = domain[code[0]]
    else:  # Otherwise, the place of a 1 in the list is the index to the domain.
        the_class = domain[code.index(1)]

    return the_class


def multi_one_hot_decode(domain, codes):
    """  One-hot decodes a list of codes.

    :param domain:  domain of possible values
    :param codes:  list of codes to decode
    :return:  list of decoded values
    """
    return [one_hot_decode(domain, code) for code in codes]


def column_domains(data_frame):
    """ Returns all the domains of the columns of a data frame as a list of lists.

    For example, if the data frame values are
    'a'  1  10
    'b'  2  20
    'c'  3  30
    'a'  1  10

    then [['a','b','c'], [1,2,3], [10,20,30]] is returned.

    :param data_frame:  data frame to extract the domains from
    :return:  the list of domains - each domain is sorted using its standard ordering
    """
    domains = {}
    for col_index in data_frame:
        domains[col_index] = sorted(data_frame[col_index].unique())
    return domains


def one_hot_code_data_frame(data_frame, domains=None):
    """ One-hot codes an entire data frame.

    For example, if the data frame values are
    a  1
    b  2
    c  3
    a  1

    then the returned data frame is
    1  0  0  1  0  0
    0  1  0  0  1  0
    0  0  1  0  0  1
    1  0  0  1  0  0

    :param data_frame:  data frame on which to operate
    :param domains:  if given, the one-hot coder uses the domains given in this list of lists, otherwise they are
        deduced from the data frame's columns
    :return:  a new one-hot coded data frame
    """
    if domains is None:
        domains = column_domains(data_frame)

    coded_matrix = []
    for row_index, row in data_frame.iterrows():
        coded_row = []
        for col_index in row.index:
            value = row[col_index]
            coded_row += one_hot_encode(value, domains[col_index])  # Using the += concatenation results in a flat list, unlike with map() or a list comprehension
        coded_matrix.append(coded_row)

    return pd.DataFrame(coded_matrix)


def split_x_y(data_frame, y_col):
    """ Splits a data frame into two frames, one containing the independent variable and the other the
     dependent variable.

    :param data_frame:  frame to be split
    :param y_col:  number of the column that holds the dependent variable
    :return:  (independent var. frame, dependent var. frame)
    """
    num_cols = data_frame.shape[1]
    x_cols = list(range(num_cols))
    x_cols.remove(y_col)

    x_data = data_frame.iloc[:, x_cols]
    y_data = data_frame.iloc[:, [y_col]]

    return x_data, y_data


def convert_indices_to_classes(classes, indices):
    """ Given a list of classes and a list of indices to that list, this function returns a list of the classes
    that correspond to the indexes.

    For example, if 'classes' is ['a', 'b', 'c'] and 'indices' is [1, 1, 3], this function returns ['a', 'a', 'c'].

    :param classes:  list of classes to be indexed
    :param indices:   list of indices
    :return:  list of classes converted from the indices
    """
    return [classes[index] for index in indices]


def normalize(matrix):
    """ Mean centers and scales for a standard deviation of 1 a numeric data matrix.

    :param matrix: data as a list of lists, i.e. matrix
    :return: none (alters data in place)
    """
    num_cols = len(matrix[0])
    num_rows = len(matrix)

    for col_index in range(num_cols):
        col = extract_col(matrix, col_index)
        col_mean = stat.mean(col)
        col_stdev = stat.stdev(col)

        for row_index in range(num_rows):
            matrix[row_index][col_index] -= col_mean
            matrix[row_index][col_index] /= col_stdev
