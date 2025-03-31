from .ml_helpers import *


def vertical_print_dict(dict):
    """ Prints a dictionary with each key and value combination on a separate line. """
    for key, value in dict.items():
        print(f"{key}: {value}")


def default_model_printer(model):
    """ Default function for pretty-printing an ML model. """
    for item in model:
        print(item)


def print_classification_prediction_comparison(actual, predicted):
    """ Prints a comparison between actual and predicted classifications and flags mismatches. """
    print("actual ---> predicted")
    for index in range(len(actual)):
        flag = "  MISMATCH!" if actual[index] != predicted[index] else ""
        print(f"{actual[index]} ---> {predicted[index]}{flag}")

    print()
    error_rate = calculate_error_rate(actual, predicted)
    print(f"Error rate: {error_rate}\n")

