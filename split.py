import metrics
import utility
import binning
import numbers
import numpy as np


def recursive_construction(dtree, X, y, col_idx, split_point, targets, limit, k, average, missing_label):

    sp = Splitter().fit(X, y)

    if sp.multiway_split(col_index=col_idx, split_points=split_point) is None:
        # print('Cannot split for this node')

        freqs = metrics.frequency(y, targets)

        dtree.add_node(X, y, col_idx, split_point, None,
                       freqs, targets, is_leaf=True)
        return

    x_parts, y_parts = sp.multiway_split(col_index=col_idx, split_points=split_point)

    for x_part, y_part in zip(x_parts, y_parts):

        one_sided_ent = metrics.one_sided_entropy(y_part, targets)
        one_sided_freqs = metrics.frequency(y_part, targets)

        if one_sided_ent == 0.0 or len(y_part) <= limit:
            dtree.add_node(x_part, y_part, col_idx, split_point, None,
                           one_sided_freqs, targets, is_leaf=True)

        else:
            be = BestEstimator(X=x_part, y=y_part, targets=targets, missing_label=missing_label)
            be.test_estimators(k=k, average=average, binary=False)

            best_est = be.get_best_estimator()

            if best_est is not None:
                split_col_idx = best_est.col_index
                second_split_point = best_est.best_point
                second_chi = best_est.best_score

                dtree.add_node(x_part, y_part, split_col_idx, second_split_point, second_chi,
                               one_sided_freqs, targets)

                recursive_construction(dtree, x_part, y_part, split_col_idx, second_split_point,
                                       targets, limit, k, average, missing_label)

            else:
                # print('Cannot split')
                dtree.add_node(x_part, y_part, None, None, None,
                               one_sided_freqs, targets, is_leaf=True)


class Splitter:
    def __init__(self):
        self.__X = None
        self.__y = None

    def __set_params(self, X, y):
        self.__X = X
        self.__y = y

    def fit(self, X, y):
        self.__set_params(X, y)
        return self

    def split(self, col_index, split_point=None):
        """
        :param split_point: expected floating number
                             if it's None, it will be a nominal split
        :param col_index: index of splitting column
        :return: split x and y parts
        """

        # Continuous split
        if split_point is not None:
            upper_condition = self.__X[:, col_index] >= split_point
            lower_condition = self.__X[:, col_index] < split_point

            upper_x, upper_y = self.__X[upper_condition], self.__y[upper_condition]
            lower_x, lower_y = self.__X[lower_condition], self.__y[lower_condition]

            x_parts = np.array([lower_x, upper_x])
            y_parts = np.array([lower_y, upper_y])

        # Nominal split, one child per nominal category
        else:
            x_parts, y_parts = [], []
            categories = np.array(list(set(self.__X[:, col_index])))

            for category in categories:
                condition = self.__X[:, col_index] == category

                x_parts.append(self.__X[condition])
                y_parts.append(self.__y[condition])

            x_parts = np.array(x_parts)
            y_parts = np.array(y_parts)

        for p in y_parts:
            if len(p) == 0:
                return None

        return x_parts, y_parts
    
    def multiway_split(self, col_index, split_points=None):

        if split_points is not None:

            if isinstance(split_points, numbers.Number):
                split_points = [split_points]

            split_points = sorted(split_points)

            x_parts, y_parts = [], []

            X = np.copy(self.__X)
            y = np.copy(self.__y)

            for i, point in enumerate(split_points):

                upper_condition = X[:, col_index] >= point
                lower_condition = X[:, col_index] < point

                # Append lower anyway
                x_parts.append(X[lower_condition])
                y_parts.append(y[lower_condition])

                if i == len(split_points) - 1:
                    x_parts.append(X[upper_condition])
                    y_parts.append(y[upper_condition])

                X = X[upper_condition]
                y = y[upper_condition]

            x_parts = np.array(x_parts)
            y_parts = np.array(y_parts)

        else:
            x_parts, y_parts = [], []
            categories = np.array(list(set(self.__X[:, col_index])))

            for category in categories:
                condition = self.__X[:, col_index] == category

                x_parts.append(self.__X[condition])
                y_parts.append(self.__y[condition])

            x_parts = np.array(x_parts)
            y_parts = np.array(y_parts)

        for p in y_parts:
            if len(p) == 0:
                return None

        return x_parts, y_parts
            
        
class BestEstimator:
    def __init__(self, X, y, targets, missing_label='?', alpha=0.05):
        self.__X = X
        self.__y = y
        self.targets = targets
        self.missing_label = missing_label
        self.alpha = alpha

        # Split points for each variable, None for nominal variables
        self._split_points = np.zeros(self.__X[0].size, dtype=object)
        # Number of attributes
        self._scores = np.zeros(self.__X[0].size)

        self.col_index = None
        self.best_point = None
        self.best_score = None

    def get_best_estimator(self):
        best_estimator_index = np.argmax(self._scores)

        self.col_index = best_estimator_index
        self.best_point = self._split_points[best_estimator_index]
        self.best_score = self._scores[best_estimator_index]

        if self.best_score == 0.0:
            return None

        return self

    def test_estimators(self, k=None, average=True, binary=False):

        for i, col in enumerate(self.__X.transpose()):

            if utility.is_numeric(x=col):
                if binary:
                    split_from, chi = self.__get_best_point(i, k, average)
                    # split_from, chi = self.__get_best_point(i, k, average, known_index, unknown_index)
                else:
                    split_from, chi = self.__get_from_binning_tree(i, k, average)
                    # split_from, chi = self.__get_from_binning_tree(i, k, average, known_index, unknown_index)
                self._split_points[i] = split_from

            else:
                chi, _ = self.__nominal_split_info(i)
                self._split_points[i] = None

            self._scores[i] = chi

    # For continuous variables
    def __get_best_point(self, col_index, k, average):

        possible_points = self._get_possible_points(col_index=col_index, k=k, average=average)
        
        if possible_points.size == 0:
            return None, 0.0

        scores = np.zeros(possible_points.size)

        for i, p in enumerate(possible_points):

            upper, lower = self.__y[self.__X[:, col_index] >= p], self.__y[self.__X[:, col_index] < p]
            # upper, lower = y[X[:, col_index] >= p], y[X[:, col_index] < p]

            if upper.size == 0 or lower.size == 0:
                chi = 0.0

            else:
                y_parts = np.array([lower, upper])

                chi, _ = metrics.calculate_chi_square(y_parts, self.targets, alpha=self.alpha)

            scores[i] = chi

        best_point_index = np.argmax(scores)

        best_point = possible_points[best_point_index]
        best_score = np.amax(scores)

        best_point = round(best_point, 2)

        return best_point, best_score

    def __get_from_binning_tree(self, col_index, k, average):
        X = np.copy(self.__X[:, col_index].reshape(-1, 1))
        y = np.copy(self.__y)

        binning_tree = binning.BinningTree(limit=0.3, k=k, average=average)
        binning_tree = binning_tree.fit(X, y)

        if binning_tree is None:
            return None, 0.0

        split_points = binning_tree.get_all_points()

        sp = Splitter().fit(X, y)

        if sp.multiway_split(0, split_points=split_points) is None:
            return split_points, 0.0

        _, y_parts = sp.multiway_split(0, split_points=split_points)

        chisq, _ = metrics.calculate_chi_square(y_parts, targets=self.targets, alpha=self.alpha)

        return split_points, chisq

    # For nominal variables
    def __nominal_split_info(self, col_index):
        # Because method 'split' expects a 2d array
        x2d = np.array([self.__X[:, col_index]]).transpose()

        splitter = Splitter().fit(x2d, self.__y)

        x_parts, y_parts = splitter.split(col_index=0)

        chi, significant = metrics.calculate_chi_square(y_parts, targets=self.targets, alpha=self.alpha)
        return chi, significant

    # Returns possible split points - for continuous variables -
    # Automatize finding k if not given
    def _get_possible_points(self, col_index, k, average):

        x = np.copy(self.__X[:, col_index])

        x.sort()
        
        if x.size == 0:
            return np.array([])
        
        if np.unique(x).size == 1:
            return np.array([])
        
        if not average:
            possible_points = np.unique(x)

        else:
            if k is None:
                unique_size = np.unique(x).size
                k = round(unique_size / 3)

            parts = np.array_split(x, k)

            possible_points = np.unique(np.array([np.mean(r) for r in parts if len(r) != 0]))

        return possible_points
