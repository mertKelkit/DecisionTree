import numbers
import numpy as np


def get_most_frequent(y):
    y = y.tolist()
    return max(set(y), key=y.count)


# Will be improved...
def is_numeric(x):
    for i in x:
        if not isinstance(i, numbers.Number) or isinstance(i, bool):
            return False
    return True


def is_numeric_array(x):
    try:
        x.astype(np.float)
    except:
        return False

    return True


def find_interval(value, points):
    # Points are expected sorted

    if isinstance(points, numbers.Number):
        points = [points]

    for i, p in enumerate(points):
        if i == 0:
            if value < p:
                return 'x < {}'.format(p)
        if i != len(points) - 1:
            if p <= value < points[i+1]:
                return '{} <= x < {}'.format(p, points[i+1])
        else:
            return 'x >= {}'.format(p)


def group_values(arr, dictionary):
    """
    *dictionary- expects a dictionary of the form:
    {
        'group1': [value1, value2, value3]
        'group2': [value4, value5, value6]
    }
    
    Example:
    {
        'summer': ['june', 'july', 'august']
        'spring': ['march', 'april', 'may']
        ...
    }
    
    * arr- expected as ndarray
    """

    for k, v in dictionary.items():
        for item in v:
            arr[arr == item] = k
            
    return arr


def any_missing(arr, missing_label='?'):
    """
    :param arr: expected 1d numpy array, numerical values
    :param missing_label: marks of missing values, e.g. 'unknown', '?'
    :return: True if include missing values, else False
    """
    if missing_label in arr:
        return True
    return False


# NO NEED
def merge_intervals(interval_str, points):
    """
    :param interval_str: type: string, each interval separated with comma e.g. 'x < 5.3,5.3 <= x < 7.6,x >= 7.6'
                                intervals may not overlap, objective is to detect overlaps
    :param points: split points, 1d array
    :return: merged intervals and rearranged split points
    """

    intervals = interval_str.split(',')

    if len(intervals) <= 1:
        return interval_str, points

    new_interval = ''

    for i, point in enumerate(points):
        possible_intervals = [i for i in intervals if str(point) in i]
