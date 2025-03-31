import random
import sys
from .math_helpers import *


def index_list(lst):
    """ Returns a list of all indices of the given list. """
    return list(range(len(lst)))


def split_list(lst, proportion_of_the_first_part):
    """ Splits a list in two according to the proportion given.

    :param lst:  list to be split
    :param proportion_of_the_first_part:  the fractional size of the first list, a value between 0 and 1
    :return:  tuple with each of the split parts
    """
    split_loc = round(len(lst) * proportion_of_the_first_part)
    split1 = lst[:split_loc]
    split2 = lst[split_loc:]

    return split1, split2


def pair_random_shuffle(lst1, lst2):
    """ Random shuffles a pair of lists together so that both are shuffled the same way.

    This function is useful for shuffling matching lists of x and y data.

    :param lst1:  first list to shuffle
    :param lst2:  second list to shuffle
    :return:  (first shuffle list, second shuffled list) tuple
    """
    indices = index_list(lst1)
    random.shuffle(indices)

    shuffled_lst1 = [lst1[index] for index in indices]
    shuffled_lst2 = [lst2[index] for index in indices]

    return shuffled_lst1, shuffled_lst2


def choose_k_uniques(lst, k):
    """ Chooses at random, k items from a list without replacement.

    :param lst:  list of items to choose from
    :param k:  number of items to choose
    :return:  list of the chosen items
    """
    not_selected_yet = list(set(lst))
    chosen = []

    for i in range(k):
        choice = random.choice(not_selected_yet)
        chosen.append(choice)
        not_selected_yet = [item for item in not_selected_yet if item != choice]  # remove all occurrences of the choice

    return chosen


def pick_indices(lst, indices):
    """ Returns the index-specified elements of a list. """
    return [lst[index] for index in indices]


def extract_col(matrix, col_number):
    """ Returns a single column of the matrix (list of lists). """
    return [row[col_number] for row in matrix]


def extract_cols(matrix, col_numbers):
    """ Returns the specified columns of a matrix as a list of lists. """
    return [pick_indices(row, col_numbers) for row in matrix]


def transpose_matrix(matrix):
    """ Transposes a list of lists so that it's columns become rows. """
    col_indexes = range(len(matrix[0]))
    columns = [[row[col_index] for row in matrix] for col_index in col_indexes]
    return columns


def determine_col_domains(matrix):
    """ Determines the domains (i.e. the values that appear within) for each column in the matrix (list of lists).

    :param matrix:  a list of lists with the data
    :return:  dictionary of attribute column numbers to domains,
        e.g. {0: ['a', 'b', 'c'], 1: ['yes', 'no'], 2: [5, 6, 7] }
    """
    domains = {}
    num_cols = len(matrix[0])
    col_numbers = list(range(num_cols))
    for col_num in col_numbers:
        attr_values = extract_col(matrix, col_num)
        domain = list(set(attr_values))
        domains[col_num] = domain
    return domains


def unique_list_with_counts(lst):
    """ Returns a tuple whose first item is a list of all the unique elements in the given list,
     and the second item is a list with the counts of the elements.

    For example, if lst = ['a','a','b','b','c','a'], the returned tuple is (['a','b','c'], [3,2,1]).
    """
    unique_lst = list(set(lst))
    counts = [lst.count(x) for x in unique_lst]
    return unique_lst, counts


def item_counts_dict(lst):
    """ Returns a list-item to count dictionary.

    For example if lst = ['a','a','b','b','c','a'], the function returns {'a': 3, 'b': 2, 'c': 1}
    """
    labels, label_counts = unique_list_with_counts(lst)
    return dict(zip(labels, label_counts))


def plurality_of_count(lst):
    """ Returns the value with the most number of appearances in the given list. """
    unique_lst, counts = unique_list_with_counts(lst)
    return unique_lst[list_argmax(counts)]


def filter_list(lst, criteria_fn):
    """ Returns only the list items that evaluate as true by the criteria function.

    :param lst:  the list to be filtered
    :param criteria_fn:  a function, f(item, index) -> True/False that takes the index and the item and returns boolean
    :return:  list containing only the items that for which criteria_fn() evaluates to True
    """
    filtered = []
    for index in range(len(lst)):
        item = lst[index]
        if criteria_fn(item, index):
            filtered.append(item)
    return filtered


def frequencies_of_list_occurrence(lst):
    """ Returns the frequency of occurrence of all the items in a list.

    :param lst:  the list to be processed
    :return:  item to frequency dictionary, e.g. {'yes': 0.2, 'no': 0.8}
    """
    total_size = len(lst)
    items, counts = unique_list_with_counts(lst)
    frequencies = {items[index]: counts[index]/total_size for index in range(len(items))}
    return frequencies


def group_by_to_dict(lst, group_label_fn):
    """ Groups a list of items in a dictionary.

    For example, if 'lst' is [1,2,3,4,5] and 'group_label_fn' is "lambda x: x%2", then
    {1: [1, 3, 5], 0: [2, 4]} is returned.

    :param lst:  the list of items to group
    :param group_label_fn:  a function that takes (list item, index of the item) as arguments and returns its group label
    :return:  dictionary of group labels to items
    """
    groups = {}

    for index in range(len(lst)):
        item = lst[index]
        group_label = group_label_fn(item, index)
        if group_label in groups.keys():
            groups[group_label].append(item)
        else:
            groups[group_label] = [item]

    return groups


def group_by_to_index_dict(lst, group_label_fn=lambda x, index: x):
    """ Groups a list of items in a dictionary whose keys are the list items and values are the indices of the items.

    For example, if 'lst' is [1, 0, 1, 0, 1] and 'group_label_fn' is "lambda x, i: x%2", then
    {1: [0, 2, 4], 0: [1, 3]} is returned.

    :param lst:  list of items to group
    :param group_label_fn:  a function that takes (list item, index of the item) as arguments and returns its group label
    :return:  dictionary of group labels to items indices
    """
    groups = {}

    for index in range(len(lst)):
        item = lst[index]
        group_label = group_label_fn(item, index)
        if group_label in groups.keys():
            groups[group_label].append(index)
        else:
            groups[group_label] = [index]

    return groups


def group_by_to_list(lst, group_label_fn=lambda x, index: x):
    """ Groups a list of items into a list of lists according to a function that labels the items..

    For example, if 'lst' is [1,2,3,4,5] and 'group_label_fn' is "lambda x, i: x%2", then
    [[1, 3, 5], [2, 4]] without guarantee of order.

    :param lst:  list of items to group
    :param group_label_fn:  a function that takes (list item, index of the item) as arguments and returns its group label
    :return:  list of lists with the groups
    """
    dict_groups = group_by_to_dict(lst, group_label_fn)
    return list(dict_groups.values())


def group_by_to_index_list(lst, group_label_fn=lambda x, _: x):
    """ Groups a list of items into a list of lists ontaining the indexes of the items.

    For example, if 'lst' is [1,2,3,4,5] and 'group_label_fn' is "lambda x, i: x%2", then
    [[0, 2, 4], [1, 3]] without guarantee of order.

    :param lst:  the list of items to group
    :param group_label_fn:  a function that takes (list item, index of the item) as arguments and returns its group label
    :return:  list of lists with the groups of item indices
    """
    dict_groups = group_by_to_index_dict(lst, group_label_fn)
    return list(dict_groups.values())


def is_list_numeric(lst):
    """ Returns True if all list items are either integer or float types. """
    for index in range(len(lst)):
        item = lst[index]
        item_type = type(item)
        if item_type is not int and item_type is not float:
            return False

    return True


def size_of_split(split):
    """ Returns the total number of items in the given split (list of lists). """
    sizes = [len(subset) for subset in split]
    return sum(sizes)


def avoid_division_by_zero(value):
    """ Avoid division by zero problems by converting zero to a small value. """
    return value if value != 0 else sys.float_info.epsilon


def sort_two_lists_on_values_of_first(lst1, lst2):
    """ Sort the first list and match the order of the second list to the first. """
    indices = index_list(lst1)
    sort_indices = sorted(indices, key=lambda i: lst1[i])
    return pick_indices(lst1, sort_indices), pick_indices(lst2, sort_indices)

