import split
import metrics
import utility
import numpy as np

'''
Created on Aug 31, 2018

@author: mertkelkit
'''


class BinningTree:
    """
    A method for choosing split points for multiway decision tree splitting
    """
    def __init__(self, limit=0.5, k=20, average=True):
        self.limit = limit
        self.k = k
        self.average = average
        self.root = None
        
    def fit(self, x, y):
        """
        :param x: expected 2d array, variable to bin
        :param y: expected 1d array, target
        """
        
        targets = list(set(y))

        be = split.BestEstimator(x, y, targets)
        be.test_estimators(k=self.k, average=self.average, binary=True)
        best_est = be.get_best_estimator()

        if best_est is not None:
            split_point = best_est.best_point

        else:
            # print('cannot bin for root')
            return None

        self.limit = self.limit * y.size

        self.root = BinNode(x, y, split_point)

        self.construct(x, y, split_point, targets)
        return self

    def construct(self, x, y, split_point, targets):

        sp = split.Splitter().fit(x, y)

        if sp.split(0, split_point=split_point) is None:
            # print('Cannot split for this node')

            self.add_node(x, y, split_point, is_leaf=True)
            return

        x_parts, y_parts = sp.split(0, split_point=split_point)

        for x_part, y_part in zip(x_parts, y_parts):

            one_sided_ent = metrics.one_sided_entropy(y_part, targets)

            if one_sided_ent == 0.0 or y_part.size <= self.limit or self.get_depth() > 1:
                self.add_node(x_part, y_part, split_point=None, is_leaf=True)

            else:
                be = split.BestEstimator(x_part, y_part, targets=targets)
                be.test_estimators(k=self.k, binary=True)
                best_est = be.get_best_estimator()

                if best_est is not None:
                    second_split_point = best_est.best_point

                    self.add_node(x_part, y_part, second_split_point)

                    self.construct(x_part, y_part, second_split_point, targets)

                else:
                    # print('Cannot split')
                    self.add_node(x_part, y_part, split_point=None, is_leaf=True)

    def add_node(self, x, y, split_point, is_leaf=False):
        self.root.insert(x, y, split_point, is_leaf)

    def get_all_points(self):
        all_points = []
        self.root.get_all_points(all_points)
        return np.sort(all_points)

    def get_depth(self):
        return self.root.get_depth()


class BinNode:
    def __init__(self, x, y, split_point, is_leaf=False):
        self.x = x
        self.y = y
        self.split_point = split_point
        self.is_leaf = is_leaf
        self.children = {}

    def insert(self, x, y, split_point, is_leaf):
        child_name = utility.find_interval(x[0, 0], self.split_point)

        if child_name not in self.children.keys():
            self.children[child_name] = BinNode(x=x, y=y, split_point=split_point, is_leaf=is_leaf)
            return

        self.children[child_name].insert(x, y, split_point, is_leaf)

    def get_all_points(self, arr):
        if self.is_leaf:
            return

        arr.append(self.split_point)

        for c in self.children.values():
            c.get_all_points(arr)

    def get_depth(self):
        depths = []

        if self is None:
            return 0

        for child in self.children.values():
            depths.append(child.get_depth())

        if len(depths) > 0:
            return max(depths) + 1

        return 0
