from .math_helpers import *
from .list_helpers import *
import statistics as stat


def train_test_split(x_data, y_data, proportion_of_training_data):
    """  Splits x and y lists of data into test and train sets according to the given proportion.

    For example if 'x_data' is [[1,2], [3,4], [5,6], [7,8] and 'y_data' is ['a', 'b', 'c', 'd'], and
    proportion_of_training_data is 0.5, this function returns
    ([[1,2], [3,4]], [[5,6], [7,8]], ['a', 'b'], ['c', 'd'])

    :param x_data:  first list (with x data)
    :param y_data:  second list (with y data)
    :param proportion_of_training_data:  relative size of the first split (i.e. training) data as a number from 0 to 1
    :return:  (x training data, x testing data, y training data, y testing data) tuple
    """
    x_train, x_test = split_list(x_data, proportion_of_training_data)
    y_train, y_test = split_list(y_data, proportion_of_training_data)
    return x_train, x_test, y_train, y_test


def random_train_test_split(x_data, y_data, proportion_of_training_data):
    """ Randomly shuffles (while preserving the match-up between the x's and y's) and splits x and y data.

    :param x_data:  first list (with x data)
    :param y_data:  second list (with y data)
    :param proportion_of_training_data:  relative size of the first split (i.e. training) data as a number from 0 to 1
    :return:  (x training data, x testing data, y training data, y testing data) tuple with shuffle data
    """
    x_data, y_data = pair_random_shuffle(x_data, y_data)
    return train_test_split(x_data, y_data, proportion_of_training_data)


def train_multi_y(ml_algorithm_fn, x_vectors, y_vectors):
    """ Trains and returns a separate model for each of the columns of 'y_vectors' using
    the same 'x_vectors'.

    :param ml_algorithm_fn:  function that takes (list of x vectors, list of y values) as arguments and returns
        a trained model
    :param x_vectors:  list of x vectors to train on
    :param y_vectors:  list of lists, where each column is to train a separate model
    :return:  list of models trained models
    """
    y_lists = transpose_matrix(y_vectors)
    models = [ml_algorithm_fn(x_vectors, y_scalar) for y_scalar in y_lists]
    return models


def serial_predict(predict_fn, x_vectors, model):
    """ Applies a prediction function to an array of feature vectors.

    :param predict_fn:  function that takes (an x feature vector, model) as arguments and returns the prediction
    :param x_vectors:  list of x vectors to predict for
    :param model:  trained model
    :return:  list of models predictions
    """
    return [predict_fn(x_vector, model) for x_vector in x_vectors]


def multi_model_winners(model_confidence_fn, models, x_vectors):
    """  Determines which model is more confident of its prediction for a list of x vectors.

    :param model_confidence_fn:  function that takes (x vector, model) as arguments and returns a measure of how
        confident the model is in its prediction (higher values indicate more confidence)
    :param models:  list of models
    :param x_vectors:  list of x vectors
    :return:  list of indexes to 'models' indicating which model has the most confidence for the corresponding row
        in 'x_vectors'
    """
    winners = []
    for x_vector in x_vectors:
        confidence_values = [model_confidence_fn(x_vector, model) for model in models]
        winner = list_argmax(confidence_values)
        winners.append(winner)
    return winners


def calculate_error_rate(actual, predicted):
    """ Calculates the prediction error rate, i.e. the fraction of incorrect predictions.

    :param actual:  list of actual classifications
    :param predicted:  list of predicted classifications
    :return:  error rate as a fraction from 0 to 1
    """
    size = len(actual)
    num_errors = 0
    for index in range(size):
        if actual[index] != predicted[index]:
            num_errors += 1

    return num_errors/size


def calculate_accuracy(actual, predicted):
    """ Calculates the prediction accuracy, i.e. the fraction of correct predictions.

    :param actual:  list of actual classifications
    :param predicted:  list of predicted classifications
    :return:  accuracy as a fraction from 0 to 1
    """
    return 1 - calculate_error_rate(actual, predicted)


def mean_squared_error(actual, predicted):
    """ Calculates the mean squared error of lists of actual and predicted values.

    :param actual:  list of actual values
    :param predicted:  list of predicted values
    :return:  mean squared error
    """
    errors = [(actual[i] - predicted[i])**2 for i in range(len(actual))]
    return stat.mean(errors)


def mean_squared_error_with_single_prediction(actual, predicted):
    """ Calculates the mean squared error of a list with a single predicted value.

    :param actual:  list of actual values
    :param predicted:  predicted value
    :return:  mean squared error
    """
    errors = [(actual[i] - predicted)**2 for i in range(len(actual))]
    return stat.mean(errors)


def mean_error(actual, predicted):
    """ Calculates the mean error of a list of value predictions.

    :param actual:  list of actual values
    :param predicted:  list of predicted values
    :return:  mean error
    """
    errors = [abs(actual[i] - predicted[i]) for i in range(len(actual))]
    return stat.mean(errors)


def calculate_learner_performance(x_train, y_train, x_test, y_test, train_fn, predict_fn, evaluate_performance_fn):
    """ A convenience function that trains a model and compares its predictions with known results.

    :param x_train:  list of x (feature) vectors for the training data
    :param y_train:  list of y (dependent variable) scalars for the training data
    :param x_test:   list of x vectors for the testing data
    :param y_test:   list of y scalars for the training data
    :param train_fn:  function that takes (x_train, y_train) arguments with training data and returns a trained model
    :param predict_fn:  function that takes (x feature vector, model) arguments and returns the prediction
    :param evaluate_performance_fn:  function that takes (known values, predictions) arguments and returns a performance
        evaluation metric, for example, fraction of correct predictions
    :return:  value of the performance metric
    """
    model = train_fn(x_train, y_train)
    predictions = serial_predict(predict_fn, x_test, model)
    return evaluate_performance_fn(y_test, predictions)


def get_labels_for_unsupervised_predictions(x_vectors, y_scalars, predict_fn, model):
    """ Determines data derived labels for the result of a clustering algorithm by picking the most frequently
    appearing label in a predicted cluster.

    :param x_vectors:  list of x (feature) vectors
    :param y_scalars:  list of labels to of those vectors
    :param predict_fn:  function that takes (x feature vector, model) arguments and returns the prediction
    :param model:  trained model of the clustering algorithm to use with the prediction function
    :return:  dictionary of {cluster number: cluster label}, e.g. {0: 'blue', 1: 'green', 2: 'red'}
    """
    # get predictions and cluster the data into a {cluster number: points in the cluster} dictionary
    training_predictions = serial_predict(predict_fn, x_vectors, model)
    indices = index_list(y_scalars)
    clusters = group_by_to_dict(indices, lambda index, _: training_predictions[index])

    label_dict = {}

    for cluster_number, cluster in clusters.items():  # loop over all clusters with their cluster number
        cluster_actual_labels = [y_scalars[index] for index in cluster]  # get all actual labels from the data for all the points in the cluster
        cluster_actual_label = list_mode(cluster_actual_labels)  # the label is the mode, i.e. the most frequently seen actual label in the cluster
        label_dict[cluster_number] = cluster_actual_label

    return label_dict


def make_cluster_list_from_labels(data, labels):
    """ Groups data by its given labels and returns the group as a list of lists.

    Example: If 'data' is ['a', 'b', 'c', 'd', 'e'] and labels is [1,0,1,1,0] then the function returns
    [['a', 'c', 'd'], ['b', 'e']]

    :param data:  list of data points to cluster
    :param labels:  list of lables that correspond to the points in order
    :return:  list of lists with the clusters
    """
    return group_by_to_list(data, lambda _, index: labels[index])


def expandable_min_search_parameterizer(param_range_increment, error_fn, increment_for_equal_errors=False):
    """ A parameterization function that returns the best parameter that resulted in the lowest value when given to
    'error_fn'.  The range of search starts with 1 to 'param_range_increment'.  If the best parameter is in the upper
    half of the range, the range is expanded by 'param_range_increment'.  The range is repeatedly expended until
    the best parameter falls within the lower half of the expanded range.

    The motivation behind this method is to overshoot the value of the best parameter found to avoid local minima in
    the search.

    :param param_range_increment:  integer by which to expand the search in each iteration
    :param error_fn:  function that take an integer parameter as an argument and returns the error measure when
        the parameter is applied
    :param increment_for_equal_errors:  set to True to keep incrementing if error is equal to the previous error
    :return:  best parameter found
    """
    min_error = MAX
    best_param = None

    # determine initial parameter range
    range_start = 1
    range_end = param_range_increment

    while True:
        for param in range(range_start, range_end + 1):  # loop through the determined parameter range
            error = error_fn(param)
            if error < min_error or (error < min_error and increment_for_equal_errors):
                min_error = error
                best_param = param

        cutoff_point = range_start + (range_end - range_start + 1)/2 - 1  # range midpoint, e.g. start=11 & end=20 ==> 11 + (20 - 11 + 1)/2 - 1 = 15
        if best_param <= cutoff_point:  # if the best parameter found is in the lower part of the parameter search range
            return best_param, min_error
        else:
            # expand the search to the next interval
            range_start = range_end + 1
            range_end = range_start + param_range_increment - 1
