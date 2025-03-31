###
# Yaron Walfish
###

from ywlib import *
import statistics as stat
import time


NUMBER_OF_FOLDS = 5
CROSS_TESTING_DATA_PROPORTION = 0.9
TUNING_PARAMETER_SEARCH_EXPANSION = 1

COMPUTER_HARDWARE_TUNING_INCREMENT = 5
FOREST_FIRES_TUNING_INCREMENT = 10
RED_WINE_QUALITY_TUNING_INCREMENT = 0.1
WHITE_WINE_QUALITY_TUNING_INCREMENT = 0.1


config = load_configuration('config.yml')


def print_classification_and_pruning_fold_experiment_results(errors, sizes, pruned_errors, pruned_sizes):
    print("----- Initial Tree -----")
    print(f"Fold Test Errors: {errors}")
    print(f"Tree Sizes: {sizes}")
    print(f"Mean Test Errors: {stat.mean(errors)} Error Std. Dev.: {stat.stdev(errors)}")
    print(f"Mean Tree Sizes: {stat.mean(sizes)} Size Std. Dev.: {stat.stdev(sizes)}")
    print()
    print("----- Pruned Tree ------")
    print(f"Fold Test Error Rates: {pruned_errors}")
    print(f"Tree Sizes: {pruned_sizes}")
    print(f"Mean Test Error Rate: {stat.mean(pruned_errors)} Error Std. Dev.: {stat.stdev(pruned_errors)}")
    print(f"Mean Tree Sizes: {stat.mean(pruned_sizes)} Size Std. Dev.: {stat.stdev(pruned_sizes)}")
    print()


def print_regression_fold_experiment_results(errors, sizes, stopping_error):
    print(f"---------- Fold Experiment - Stopping Error = {stopping_error} ---------")
    print(f"Fold Test Errors: {errors}")
    print(f"Tree Sizes: {sizes}")
    print(f"Mean Test Error: {stat.mean(errors)} Error Std. Dev.: {stat.stdev(errors)}")
    print(f"Mean Tree Sizes: {stat.mean(sizes)} Size Std. Dev.: {stat.stdev(sizes)}")
    print()


def print_stopping_error_tuning_experiment_result(error, size, parameter, stopping_error):
    print(f"---------- Tuning Iteration {parameter}: Stopping Error = {stopping_error} ---------")
    print(f"Validation Error: {error}")
    print(f"Tree Size: {size}")
    print()


def print_stopping_error_tuning_final_result(best_param, best_stopping_error, best_validation_error):
    print(f">>>------> Stopping Error Tuning Final Result <------<<<")
    print(f"Best Stopping Error: {best_stopping_error}")
    print(f"Found on Iteration: {best_param}")
    print(f"Best Tuning Validation Error: {best_validation_error}")
    print()


def print_title(title):
    """ Title print helper. """
    print()
    print()
    print(f"*** {title} ***")
    print()


def print_divider():
    print()
    print()
    print("----------------------------------------------------------------------------------------------------")


def create_single_classification_and_pruning_experiment_fn(x_prune_data, y_prune_data):
    """ Returns a function that runs a single decision tree classification and pruning experiment.

    :param x_prune_data:   list of x (feature) data for pruning
    :param y_prune_data:   list of dependent variables data for pruning
    :return: :return:  function that takes (x_train, x_test, y_train, y_test) as arguments and returns
        (error of full tree on test data, full tree size, error of pruned tree on test data, pruned tree size) tuple
    """
    def experiment(x_train, x_test, y_train, y_test):
        # train full tree and get measurements
        tree = decision_tree_train_classification(x_train, y_train)
        size = tree.size()
        predictions = serial_predict(decision_tree_predict, x_test, tree)
        error = calculate_error_rate(y_test, predictions)

        # prune tree and get measurements
        tree.prune(x_prune_data, y_prune_data)
        predictions = serial_predict(decision_tree_predict, x_test, tree)
        pruned_error = calculate_error_rate(y_test, predictions)
        pruned_size = tree.size()

        return error, size, pruned_error, pruned_size

    return experiment


def classification_and_pruning_fold_experiment(x_cross_data, x_prune_data, y_cross_data, y_prune_data):
    """  Runs a fold experiment for decision tree classification and pruning and prints the results.

    :param x_cross_data:  list of x (feature) data for cross training
    :param x_prune_data:  list of x (feature) data for pruning
    :param y_cross_data:  list of dependent variables data for cross training
    :param y_prune_data:  list of dependent variables data for pruning
    :return: n/a
    """
    results = iterate_over_stratified_folds(x_cross_data, y_cross_data, NUMBER_OF_FOLDS,
                                            create_single_classification_and_pruning_experiment_fn(x_prune_data, y_prune_data))

    errors = extract_col(results, 0)
    sizes = extract_col(results, 1)
    pruned_errors = extract_col(results, 2)
    pruned_sizes = extract_col(results, 3)

    print_classification_and_pruning_fold_experiment_results(errors, sizes, pruned_errors, pruned_sizes)


def create_single_regression_experiment_fn(stopping_error):
    """ Returns a function that runs a single decision tree regression experiment.

    :param stopping_error:  stopping error to use for early stopping
    :param y_prune_data:  list of dependent variables data for pruning
    :return: :return:  function that takes (x_train, x_test, y_train, y_test) as arguments and returns
        (error of tree on test data, tree size) tuple
    """
    def experiment(x_train, x_test, y_train, y_test):
        tree = decision_tree_train_regression(x_train, y_train, stopping_error)
        size = tree.size()
        predictions = serial_predict(decision_tree_predict, x_test, tree)
        error = mean_squared_error(y_test, predictions)

        return error, size

    return experiment


def regression_fold_experiment(x_cross_data, y_cross_data, stopping_error=None):
    """  Runs a fold experiment for decision tree regression and prints the results.

        :param x_cross_data:  list of x (feature) data for cross training
        :param y_cross_data:  list of dependent variables data for cross training
        :param stopping_error:  stopping error to use for early stopping
        :return: n/a
    """
    results = iterate_over_folds(x_cross_data, y_cross_data, NUMBER_OF_FOLDS, create_single_regression_experiment_fn(stopping_error))

    errors = extract_col(results, 0)
    sizes = extract_col(results, 1)

    print_regression_fold_experiment_results(errors, sizes, stopping_error)


def convert_param_to_stopping_error(parameter, stopping_error_increment):
    """ Converts the 1, 2, 3, ... sequence given by the parameterizer into the stopping error parameter being tuned. """
    return (parameter) * stopping_error_increment


def create_single_regression_stopping_error_tuning_experiment_fn(x_train, x_validate, y_train, y_validate, stopping_error_increment):
    """

    :param x_train:  list of x (feature) data for training
    :param x_validate:  list of x (feature) data for validating
    :param y_train:  list of dependent variables data for training
    :param y_validate: list of dependent variables data for validating
    :param stopping_error_increment:  increment of stopping error value to use when tuning
    :return:  the experiment function configured by a sequential integer starting with 1
        (which the function converts to the stopping error) and returns the error for the parameter under test
    """
    def experiment(parameter):
        stopping_error = convert_param_to_stopping_error(parameter, stopping_error_increment)  # convert the parameter to the stopping error
        tree = decision_tree_train_regression(x_train, y_train, stopping_error)
        size = tree.size()
        predictions = serial_predict(decision_tree_predict, x_validate, tree)
        error = mean_squared_error(y_validate, predictions)

        print_stopping_error_tuning_experiment_result(error, size, parameter, stopping_error)

        return error

    return experiment


def tune_regression_stopping_error_parameter(x_train, x_validate, y_train, y_validate, stopping_error_increment):
    """ Runs a stopping error tuning experiment, prints the result, and returns the tuned stopping error.

    :param x_train:  list of x (feature) data for training
    :param x_validate:  list of x (feature) data for validating
    :param y_train:  list of dependent variables data for training
    :param y_validate: list of dependent variables data for validating
    :param stopping_error_increment:  increment of stopping error value to use when tuning
    :return:  resulting stopping error
    """
    parameter_error_fn = create_single_regression_stopping_error_tuning_experiment_fn(x_train, x_validate, y_train, y_validate, stopping_error_increment)
    best_param, best_validation_error = expandable_min_search_parameterizer(TUNING_PARAMETER_SEARCH_EXPANSION, parameter_error_fn, True)
    best_stopping_error = convert_param_to_stopping_error(best_param, stopping_error_increment)

    print_stopping_error_tuning_final_result(best_param, best_stopping_error, best_validation_error)

    return best_stopping_error


start_time = time.process_time()


# Abalone
print_divider()

x_data, y_data = load_abalone_data(config['abalone_dataset_file'])
x_data, y_data = pair_random_shuffle(x_data, y_data)

print_title("Abalone Dataset - Classification Tree and Pruning")
x_cross_data, x_prune_data, y_cross_data, y_prune_data = train_test_split(x_data, y_data, CROSS_TESTING_DATA_PROPORTION)
classification_and_pruning_fold_experiment(x_cross_data, x_prune_data, y_cross_data, y_prune_data)


# Car Evaluation
print_divider()

x_data, y_data = load_car_evaluation_data(config['car_evaluation_dataset_file'])
x_data, y_data = pair_random_shuffle(x_data, y_data)

print_title("Car Evaluation Dataset - Classification Tree and Pruning")
x_cross_data, x_prune_data, y_cross_data, y_prune_data = train_test_split(x_data, y_data, CROSS_TESTING_DATA_PROPORTION)
classification_and_pruning_fold_experiment(x_cross_data, x_prune_data, y_cross_data, y_prune_data)


# Image Segmentation
print_divider()

x_data, y_data = load_segmentation_data(config['segmentation_dataset_file'])
x_data, y_data = pair_random_shuffle(x_data, y_data)

print_title("Image Segmentation Dataset - Classification Tree and Pruning")
x_cross_data, x_prune_data, y_cross_data, y_prune_data = train_test_split(x_data, y_data, CROSS_TESTING_DATA_PROPORTION)
classification_and_pruning_fold_experiment(x_cross_data, x_prune_data, y_cross_data, y_prune_data)


# Computer Hardware
print_divider()

x_data, y_data = load_computer_hardware_data(config['computer_hardware_dataset_file'])
x_data, y_data = pair_random_shuffle(x_data, y_data)

print_title("Computer Hardware Dataset - Regression Tree and Early Stopping")
x_cross_data, x_validate_data, y_cross_data, y_validate_data = train_test_split(x_data, y_data, CROSS_TESTING_DATA_PROPORTION)

best_stopping_error = tune_regression_stopping_error_parameter(x_cross_data, x_validate_data, y_cross_data, y_validate_data, COMPUTER_HARDWARE_TUNING_INCREMENT)  # use all of the cross testing data as the training data for tuning
regression_fold_experiment(x_cross_data, y_cross_data)  # run experiment without early stopping
regression_fold_experiment(x_cross_data, y_cross_data, best_stopping_error)


# Forest Fires
print_divider()

x_data, y_data = load_forestfires_data(config['forestfires_dataset_file'])
x_data, y_data = pair_random_shuffle(x_data, y_data)

print_title("Forest Fires Dataset - Regression Tree and Early Stopping")
x_cross_data, x_validate_data, y_cross_data, y_validate_data = train_test_split(x_data, y_data, CROSS_TESTING_DATA_PROPORTION)

best_stopping_error = tune_regression_stopping_error_parameter(x_cross_data, x_validate_data, y_cross_data, y_validate_data, FOREST_FIRES_TUNING_INCREMENT)  # use all of the cross testing data as the training data for tuning
regression_fold_experiment(x_cross_data, y_cross_data)  # run experiment without early stopping
regression_fold_experiment(x_cross_data, y_cross_data, best_stopping_error)


# Red Wine Quality
print_divider()

x_data, y_data = load_wine_quality_data(config['red_wine_quality_dataset_file'])
x_data, y_data = pair_random_shuffle(x_data, y_data)

print_title("Red Wine Quality Dataset - Regression Tree and Early Stopping")
x_cross_data, x_validate_data, y_cross_data, y_validate_data = train_test_split(x_data, y_data, CROSS_TESTING_DATA_PROPORTION)

best_stopping_error = tune_regression_stopping_error_parameter(x_cross_data, x_validate_data, y_cross_data, y_validate_data, RED_WINE_QUALITY_TUNING_INCREMENT)  # use all of the cross testing data as the training data for tuning
regression_fold_experiment(x_cross_data, y_cross_data)  # run experiment without early stopping
regression_fold_experiment(x_cross_data, y_cross_data, best_stopping_error)


# White Wine Quality
print_divider()

x_data, y_data = load_wine_quality_data(config['white_wine_quality_dataset_file'])
x_data, y_data = pair_random_shuffle(x_data, y_data)

print_title("White Wine Quality Dataset - Regression Tree and Early Stopping")
x_cross_data, x_validate_data, y_cross_data, y_validate_data = train_test_split(x_data, y_data, CROSS_TESTING_DATA_PROPORTION)

best_stopping_error = tune_regression_stopping_error_parameter(x_cross_data, x_validate_data, y_cross_data, y_validate_data, WHITE_WINE_QUALITY_TUNING_INCREMENT)  # use all of the cross testing data as the training data for tuning
regression_fold_experiment(x_cross_data, y_cross_data)  # run experiment without early stopping
regression_fold_experiment(x_cross_data, y_cross_data, best_stopping_error)


end_time = time.process_time()
elapsed_min = (end_time - start_time) / 60

print()
print(f">>>>> Total elapsed time in minutes: {elapsed_min} <<<<<")
