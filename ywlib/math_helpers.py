import math
from .list_helpers import *


MIN = -sys.maxsize - 1
MAX = sys.maxsize


def list_argmax(lst):
    """ Returns the index of the largest element in a list. """
    return lst.index(max(lst))


def list_argmin(lst):
    """ Returns the index of the largest element in a list. """
    return lst.index(min(lst))


def dict_argmax(dct):
    """ Returns the key of the largest value in a dictionary. """
    return max(dct, key=dct.get)


def dot_product(vect1, vect2):
    """ Returns the dot product of two vectors. """
    return sum([vect1[index]*vect2[index] for index in range(len(vect1))])


def squared_distance(vect1, vect2):
    """ Returns the squared Euclidean distance between two vectors. """
    return sum([(vect1[index] - vect2[index])**2 for index in range(len(vect1))])


def euclidean_distance(vect1, vect2):
    """ Returns the Euclidean distance between two vectors. """
    return math.sqrt(squared_distance(vect1, vect2))


def sum_vectors(vectors):
    """ Returns the sum of a list of vectors. """
    return [sum(component) for component in zip(*vectors)]


def normalize_distribution(distribution):
    """ Given a distribution as a dictionary of items to probabilities, this function returns the same but
    with the probabilities normalized so that they total 1.
    """
    normalizer = sum(distribution.values())
    return {key: value/normalizer for key, value in distribution.items()}


def normalize_values(values):
    """ Given a list of numbers, this function returns a normalized list,
     where each number is divided by the total sum.
    """
    normalizer = sum(values)
    return [x/normalizer for x in values]


def list_mode(lst):
    """ Returns the statistical mode of the list.  Unlike the mode function in the Python statistics package,
    if there are equal maximum counts, it returns one of the values instead of failing. """
    items, counts = unique_list_with_counts(lst)
    index = list_argmax(counts)
    return items[index]


