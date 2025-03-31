###
# Yaron Walfish
###

from copy import deepcopy
from .ml_helpers import *

MAX_DOMAIN_SIZE_FOR_DISCRETE_NUMERIC_ATTRIBUTE = 30


def calculate_entropy(probabilities):
    """ Calculates the entropy from a list of probabilities or frequency counts. """
    n = sum(probabilities)  # normalizing factor for the probabilities
    entropy_terms = [-(p/n) * math.log2(p/n) for p in probabilities]
    return sum(entropy_terms)


def entropy_of_list(lst):
    """ Calculates the entropy of a list of values by considering their frequencies in the list. """
    _, counts = unique_list_with_counts(lst)
    return calculate_entropy(counts)


def expected_entropy_of_split(split):
    """ Returns the expected entropy of a split, given as a list of lists.

    :param split:  list of lists
    :return:  the entropy
    """
    n = size_of_split(split)  # total number of elements in all lists
    entropies = [(len(subset)/n) * entropy_of_list(subset) for subset in split]
    return sum(entropies)


def information_value_of_split(split):
    """ Returns information value of a split, given as a list of lists.

    :param split:  list of lists
    :return:  information value
    """
    counts = [len(subset) for subset in split]
    return calculate_entropy(counts)


def calculate_split_points(values, classifications):
    """ Returns a list of candidate split points by taking the midpoints between all adjacent data points whose
    classification is different.

    :param values:  attribute values
    :param classifications:  classifications corresponding to those attribute values in the data point
    :return:  value of the split point
    """
    if len(values) < 2:
        return []

    split_pints = []
    classifications, values = sort_two_lists_on_values_of_first(classifications, values)  # sort by y while matching the x order

    for index in range(0, len(values) - 1):
        if classifications[index] != classifications[index + 1]:  # only add split-points between two data points that have different y values
            midpoint = (values[index] + values[index + 1]) / 2
            split_pints.append(midpoint)

    return split_pints


def is_numeric_attribute(values):
    """ A heuristic to determine if an attribute is continuous numeric or categorical.

    :param values:  attribute values to base the determination on
    :return:  True if numeric, False if not
    """
    if is_list_numeric(values):
        domain, _ = unique_list_with_counts(values)
        if len(domain) <= MAX_DOMAIN_SIZE_FOR_DISCRETE_NUMERIC_ATTRIBUTE:
            return False

        return True
    else:
        return False


def get_attributes_metadata(x_vectors):
    """ Return a dictionary where the attribute indices are the keys and the values are True/False depending on
    whether the attribute is numeric/categorical.

    :param x_vectors:  list of x (feature) vectors
    :return:  attribute dictionary, e.g. { 0: True, 1: False, 2: True }
    """
    attribute_indices = index_list(x_vectors[0])

    metadata = {}
    for index in attribute_indices:
        values = extract_col(x_vectors, index)
        metadata[index] = is_numeric_attribute(values)

    return metadata


def remove_attribute(attributes, attribute_to_remove):
    """ Removes a single attribute from the attributes data structure.

    :param attributes:  attributes data structure
    :param attribute_to_remove:  attribute to remove
    :return:  a deep copied attributes data structure with the attribute removed
    """
    attributes = deepcopy(attributes)
    attributes.pop(attribute_to_remove)
    return attributes


def homogeneous(lst):
    """ If the given list is made of a single value, that value is returned, otherwise the function returns None. """
    unique = set(lst)
    return list(unique)[0] if len(unique) == 1 else None


def gain_ratio_of_categorical_attribute_split(values, classifications, pre_split_entropy):
    """ Returns the gain ratio for a categorical attribute, which is split on each of the categories.

    :param values:  attribute values
    :param classifications:  classifications corresponding to those attribute values in the data point
    :param pre_split_entropy:  entropy before the split was done
    :return:  gain ratio of the attribute split
    """
    split = group_by_to_list(classifications, lambda x, i: values[i])
    gain = pre_split_entropy - expected_entropy_of_split(split)
    information_value = information_value_of_split(split)

    return gain/avoid_division_by_zero(information_value)


def gain_ratio_of_best_numeric_attribute_split(values, classifications, pre_split_entropy):
    """ Returns the highest gain ration and best split value for a numeric attribute.

    :param values:  attribute values
    :param classifications:  classifications corresponding to those attribute values in the data point
    :param pre_split_entropy:  entropy before the split was done
    :return:  (gain ratio of the attribute split, split value)
    """
    split_values = calculate_split_points(values, classifications)  # list of all midpoints between the values
    gain_ratios = []

    for split_value in split_values:  # for all split values
        # calculate gain ration for the split
        split = group_by_to_list(classifications, lambda x, index: values[index] < split_value)
        gain = pre_split_entropy - expected_entropy_of_split(split)
        information_value = information_value_of_split(split)
        gain_ratio = gain/avoid_division_by_zero(information_value)
        gain_ratios.append(gain_ratio)

    best_index = list_argmax(gain_ratios)  # index of split value with the best gain ratio

    return gain_ratios[best_index], split_values[best_index]


def mean_squared_error_of_split(split):
    """ Returns the mean squared error of all values in a split, where the deviation of each value is from the
    mean of its own subset.

    :param split:  split as a list of lists
    :return:  mean squared error
    """
    actual = []
    predicted = []

    for subset in split:  # for each subset in the split
        actual += subset  # this flattens the split into one list
        predicted += [stat.mean(subset)]*len(subset)  # for each actual value, the prediction is the mean of its subset

    return mean_squared_error(actual, predicted)


def mean_squared_error_of_best_numeric_attribute_split(x_scalars, y_scalars):
    """ Returns the lowest mean squared error after the split with the best split value for a continuous attribute.

    :param x_scalars:
    :param y_scalars:
    :return:
    """
    split_values = calculate_split_points(x_scalars, y_scalars)
    errors = []

    for split_value in split_values:  # for all split values
        split = group_by_to_list(y_scalars, lambda x, index: x_scalars[index] < split_value)  # make the binary split
        error = mean_squared_error_of_split(split)
        errors.append(error)

    best_index = list_argmin(errors)  # index of lowest error and consequently best split

    return errors[best_index], split_values[best_index]


def select_best_attribute_for_classification(x_vectors, y_scalars, attributes):
    """ Selects the attribute with the maximum gain ratio among the given attributes.

    :param x_vectors:  list of x (feature) vectors
    :param y_scalars:  list of dependent variables
    :param attributes:   attributes metadata
    :return:  the best attribute (index column number of the data)
    """
    pre_split_entropy = entropy_of_list(y_scalars)
    gain_ratios = []
    split_points = []
    attribute_indices = list(attributes.keys())

    for index in attribute_indices:  # for each attribute
        attribute_values = extract_col(x_vectors, index) # get all attribute values from the data

        is_continuous = attributes[index]
        if is_continuous:
            # get continuous numeric attribute's best gain ration with its best split
            gain_ratio, split_point = gain_ratio_of_best_numeric_attribute_split(attribute_values, y_scalars, pre_split_entropy)
        else:
            # get categorical attributes gain ratio
            gain_ratio = gain_ratio_of_categorical_attribute_split(attribute_values, y_scalars, pre_split_entropy)
            split_point = None

        gain_ratios.append(gain_ratio)
        split_points.append(split_point)

    # select the attribute with the highest gain ratio and take the split point for the numeric attributes
    best_attribute_location = list_argmax(gain_ratios)
    best_attribute = attribute_indices[best_attribute_location]
    best_split_point = split_points[best_attribute_location]

    return best_attribute, best_split_point


def select_best_attribute_for_regression(x_vectors, y_scalars):
    """ Selects the attribute with the least mean squared error for its best split.

   :param x_vectors:  list of x (feature) vectors
   :param y_scalars:  list of dependent variables
   :return:  the best attribute (index column number of the data)
   """
    errors = []
    split_points = []
    attribute_indices = index_list(x_vectors[0])

    for index in attribute_indices:  # for each attribute
        attribute_values = extract_col(x_vectors, index)  # get all attribute values from the data

        # determine the best split and lowest error for this attribute
        error, split_point = mean_squared_error_of_best_numeric_attribute_split(attribute_values, y_scalars)
        errors.append(error)
        split_points.append(split_point)

    # select the attribute with the lowest error and take note of its split point
    best_attribute_location = list_argmin(errors)
    best_attribute = attribute_indices[best_attribute_location]
    best_split_point = split_points[best_attribute_location]

    return best_attribute, best_split_point


class DecisionTree:
    """ A data class that models the decision tree. """

    def __init__(self, default_prediction=None):
        self.prediction = default_prediction  # this node's prediction if a leaf or if the subtrees are inaccessible
        self.node_attribute_index = None  # index (to the x vector) of the feature tested in this node
        self.split_point = None  # holds the split point if the attribute is continuous
        self.children = {}  # {decision value: to subtree} dictionary to hold this nodes children

    def predict(self, query):
        """ Makes a prediction from the tree.

        :param query:  data point to classify
        :return:  the prediction
        """
        if len(self.children) == 0:  # if leaf node
            return self.prediction

        attribute_value = query[self.node_attribute_index]
        if self.split_point:  # if numeric attribute
            branch_value = attribute_value < self.split_point  # True or False depending on the comparison
        else:  # categorical
            branch_value = attribute_value

        if branch_value in self.children:
            decision_subtree = self.children[branch_value]
            return decision_subtree.predict(query)
        else:
            return self.prediction  # no subtree matches the attribute's value, therefore return the default prediction

    def train_classification(self, x_vectors, y_scalars, attribute_metadata=None):
        """ Creates a classification decision tree from the given data.  Handles both continuous numeric and
        categorical attributes.

        :param x_vectors:  list of x (feature) vectors
        :param y_scalars:  list of dependent variables
        :param attribute_metadata:   information about the attributes this tree is using
        :return:  a trained tree
        """
        if not attribute_metadata:  # initialize attribute metadata
            attribute_metadata = get_attributes_metadata(x_vectors)

        single_y = homogeneous(y_scalars)
        if single_y is not None:  # recursion terminating condition: if all classifications are the same
            self.prediction = single_y  # predict that single classification
            return self

        self.prediction = plurality_of_count(y_scalars)  # used in non-leaf nodes in case the a previously unseen value is encountered

        if len(attribute_metadata) == 0:  # recursion terminating condition: if no attributes left, predict the plurality category
            return self

        self.node_attribute_index, self.split_point = select_best_attribute_for_classification(x_vectors, y_scalars, attribute_metadata)

        attribute_values = extract_col(x_vectors, self.node_attribute_index)

        if self.split_point:  # numeric attribute
            attribute_index_split_dict = group_by_to_index_dict(y_scalars, lambda x, i: attribute_values[i] < self.split_point)
            subtree_attributes_metadata = attribute_metadata
        else:  # categorical attribute
            attribute_index_split_dict = group_by_to_index_dict(y_scalars, lambda x, i: attribute_values[i])
            subtree_attributes_metadata = remove_attribute(attribute_metadata, self.node_attribute_index)

        branch_values = attribute_index_split_dict.keys()  # comparison results that lead to the subtrees

        # create child subtrees
        for branch_value in branch_values:
            x_vectors_subset = pick_indices(x_vectors, attribute_index_split_dict[branch_value])
            y_scalars_subset = pick_indices(y_scalars, attribute_index_split_dict[branch_value])

            subtree = DecisionTree()
            subtree.train_classification(x_vectors_subset, y_scalars_subset, subtree_attributes_metadata)
            self.children[branch_value] = subtree

        return self

    def train_regression(self, x_vectors, y_scalars, stopping_error=None):
        """ Creates a regression decision tree from the given data.  Handles continuous numeric attributes only.

        :param x_vectors:  list of x (feature) vectors
        :param y_scalars:  list of dependent variables
        :param stopping_error:  when a branch's error is below this threshold, it is no longer expanded -
            pass None ot deactivate this feature
        :return:  a trained tree
        """
        single_y = homogeneous(y_scalars)
        if single_y is not None:  # recursion terminating condition: if all responses are the same
            self.prediction = single_y  # predict that single response
            return self

        self.prediction = stat.mean(y_scalars)

        # implement early stopping
        if stopping_error:
            mse = mean_squared_error_with_single_prediction(y_scalars, self.prediction)
            if mse <= stopping_error:  # if node error is no more than the stopping error
                return self  # cut off the branches and predict the mean (which was already stored in self.prediction)

        self.node_attribute_index, self.split_point = select_best_attribute_for_regression(x_vectors, y_scalars)
        attribute_values = extract_col(x_vectors, self.node_attribute_index)
        attribute_index_split_dict = group_by_to_index_dict(y_scalars, lambda x, i: attribute_values[i] < self.split_point)

        branch_values = attribute_index_split_dict.keys()  # comparison results that lead to the subtrees (True/False branches)

        if len(branch_values) == 1:  # recursion terminating condition: if there is only one branch, then the child tree is identical to this tree
            return self  # therefore, stop the recursion, and predict the mean

        # create child subtrees
        for branch_value in branch_values:
            x_vectors_subset = pick_indices(x_vectors, attribute_index_split_dict[branch_value])
            y_scalars_subset = pick_indices(y_scalars, attribute_index_split_dict[branch_value])

            subtree = DecisionTree()
            subtree.train_regression(x_vectors_subset, y_scalars_subset, stopping_error)
            self.children[branch_value] = subtree

        return self

    def prune(self, x_vectors, y_scalars, root_tree=None, base_error=None):
        """ Performs a reduced-error pruning on the tree.

        :param x_vectors:  list of x (feature) vectors of validation data
        :param y_scalars:  list of dependent variables of validation data
        :param root_tree:  the root of the tree pruned (used in calculating the performance of the pruned tree)
        :param base_error:  error to beat when pruning
        :return:
        """
        if not root_tree:
            root_tree = self  # initialize root tree

        if not base_error:  # establish a baseline error for the original tree
            predictions = serial_predict(decision_tree_predict, x_vectors, root_tree)
            base_error = calculate_error_rate(y_scalars, predictions)

        if len(self.children) == 0:  # terminate if leaf node
            return

        # save children and prune node
        saved_children = self.children
        self.children = {}

        # calculate error for the pruned root tree
        predictions = serial_predict(decision_tree_predict, x_vectors, root_tree)
        error = calculate_error_rate(y_scalars, predictions)

        if error <= base_error:  # if pruned tree is better
            return  # keep node pruned
        else:
            self.children = saved_children  # restore subtrees
            for child in self.children.values():  # prune all children
                child.prune(x_vectors, y_scalars, root_tree, base_error)

    def size(self):
        """ Returns the node count of the tree. """
        if len(self.children) == 0:  # leaf node
            return 1

        sub_counts = [child.size() for child in self.children.values()]
        return sum(sub_counts) + 1


def decision_tree_train_classification(x_vectors, y_scalars):
    """ Trains a decision tree on classification data.

    :param x_vectors:  list of x (feature) vectors
    :param y_scalars:  list of dependent variables
    :return:  trained decision tree
    """
    tree = DecisionTree()
    tree.train_classification(x_vectors, y_scalars)
    return tree


def decision_tree_train_regression(x_vectors, y_scalars, stopping_error=None):
    """ Trains a decision tree on a regression data.

    :param x_vectors:  list of x (feature) vectors
    :param y_scalars:  list of dependent variables
    :return:  trained decision tree
    """
    tree = DecisionTree()
    tree.train_regression(x_vectors, y_scalars, stopping_error)
    return tree


def decision_tree_predict(query, tree):
    """ Wraps the decision tree prediction function to create an interface compatible with ML helper functions in this library.

    :param query:  feature vector to make a prediction for
    :param tree:  the trained tree
    :return:  the prediction
    """
    return tree.predict(query)





