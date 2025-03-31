from .list_helpers import *


def fold_split(data, test_fold_number, total_folds):
    """ Splits the data into folds and returns the indicated fold as the testing data, and the rest of the data as
    the training data.

    :param data:  all the data as a list of lists
    :param test_fold_number:  index number of the fold to return as the testing data (zero indexed)
    :param total_folds:  total number of folds to split the data into
    :return:  (training data, testing data) tuple
    """
    len_test = math.floor(len(data) / total_folds)

    # calculate the index boundaries of the training data split list
    test_start = len_test * test_fold_number
    test_end = test_start + len_test

    test_data = data[test_start:test_end]
    train_data = data[:test_start] + data[test_end:]

    return train_data, test_data


def stratified_fold_split(x_data, y_data, test_fold_number, total_folds):
    """ Splits classification data info folds in which the proportions of the class labels represented remain
    the same as they are in the whole dataset (within the restrictions of integral division) and returns
    the training and testing split for the specified fold.

    Example: x_data = [10, 21, 22, 13, 14, 15, 16, 27, 18]
             y_data = [1, 2, 2, 1, 1, 1, 1, 2, 1]
             stratified_fold_split(x_data, y_data, 1, 3) returns ([10, 21, 13, 16, 27, 18], [1, 2, 1, 1, 2, 1], [22, 14, 15], [2, 1, 1])

    :param x_data:  list of x (feature) vectors
    :param y_data:  list of classifications
    :param test_fold_number:  index number of the fold to return as the testing data (zero indexed)
    :param total_folds:  total number of folds to split the data into
    :return:  (x training data, x testing data, y training data, x testing data) tuple
    """
    layers = group_by_to_index_list(y_data)  # group the data into its class labels, where each group is a list with indices to the original data

    # lists that hold the indices of the data points in the splits
    complete_train_indices = []
    complete_test_indices = []

    # do a training/testing split for each of the class groups and add those to the overall training and testing sets
    for layer in layers:
        layer_train_indices, layer_test_indices = fold_split(layer, test_fold_number, total_folds)
        complete_train_indices += layer_train_indices
        complete_test_indices += layer_test_indices

    # The following lines sort the indices to retain the original relative ordering in the data.  This is so that
    # within each of the training and testing splits, if a point appeared before another in the original data,
    # it would do so within the split.
    complete_train_indices.sort()
    complete_test_indices.sort()

    # assemble the index lists into lists of the actual data
    x_train = pick_indices(x_data, complete_train_indices)
    y_train = pick_indices(y_data, complete_train_indices)
    x_test = pick_indices(x_data, complete_test_indices)
    y_test = pick_indices(y_data, complete_test_indices)

    return x_train, x_test, y_train, y_test


def iterate_over_folds(x_data, y_data, total_folds, experiment_fn):
    """ Splits the given data into training/testing folds and repeats an experiment, given by a functional argument,
    for each of the splits.

    :param x_data:  list of x (feature) vectors
    :param y_data:  list of dependent variables
    :param total_folds:  total number of folds to split the data into
    :param experiment_fn:  a function that takes (x training data, x testing data, y training data, x testing data) as
        arguments and returns a result
    :return:  list of the results returned by each of the experiments
    """
    results = []
    for test_fold_number in range(total_folds):
        x_train, x_test = fold_split(x_data, test_fold_number, total_folds)
        y_train, y_test = fold_split(y_data, test_fold_number, total_folds)
        result = experiment_fn(x_train, x_test, y_train, y_test)
        results.append(result)

    return results


def iterate_over_stratified_folds(x_data, y_data, total_folds, experiment_fn):
    """ Splits the given classification data into stratified training/testing folds and repeats an experiment,
    given by a functional argument, for each of the splits.

    :param x_data:  list of x (feature) vectors
    :param y_data:  list of dependent variables
    :param total_folds:  total number of folds to split the data into
    :param experiment_fn:  a function that takes (x training data, x testing data, y training data, x testing data) as
        arguments and returns its result
    :return: list of the results returned by each of the experiments
    """
    results = []
    for test_fold_number in range(total_folds):
        x_train, x_test, y_train, y_test = stratified_fold_split(x_data, y_data, test_fold_number, total_folds)
        result = experiment_fn(x_train, x_test, y_train, y_test)
        results.append(result)

    return results
