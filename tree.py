import random

from split import *


class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
        self.number_of_leaves = 0

    def construct_root(self, col_idx, split_point, chi, frequencies, targets):
        if len(targets) > len(Node.target_colors):
            Node.generate_colors(len(targets) - len(Node.target_colors))
        self.root = Node(col_idx=col_idx, split_point=split_point, chi=chi,
                         frequencies=frequencies, targets=targets)

    def fit(self, X, y, k=20, average=True, limit_percentage=0.05, missing_label='?', alpha=0.05):

        self.__setattr__('missing_label', missing_label)

        targets = list(set(y))

        limit = round(limit_percentage * len(y))

        root_freqs = metrics.frequency(y, targets)

        be = BestEstimator(X, y, targets, missing_label, alpha)
        be.test_estimators(k=k, average=average, binary=False)
        best_est = be.get_best_estimator()
        
        if best_est is None:
            print('Cannot split for root')
            return None

        col_idx = best_est.col_index
        split_point = best_est.best_point
        chi = best_est.best_score

        self.construct_root(col_idx=col_idx, split_point=split_point, chi=chi,
                            frequencies=root_freqs, targets=targets)

        # Recursive construction of tree
        recursive_construction(dtree=self, X=X, y=y, col_idx=col_idx, split_point=split_point,
                               targets=targets, limit=limit, k=k, average=average, missing_label=missing_label)

        self.root.prune(limit, alpha)

        self.root.merge_results(alpha)

        self.count_leaves()

        return self

    def predict(self, X):
        predictions = list(map(lambda x: self.root.predict(x), X))
        return predictions

    def add_node(self, X, y, col_idx, split_point, chi, freq, targets, is_leaf=False, missing=False, missing_label='?'):
        if self.root is None:
            print('Error, no root is found. Call method "construct_root"')
            return
        self.root.insert(X, y, col_idx, split_point, chi, freq, targets, is_leaf)

    def count_node(self):
        return self.root.count()

    def get_depth(self):
        return self.root.depth()

    def count_leaves(self):
        self.number_of_leaves = self.root.count_leaves()

    def get_all_split_points(self):
        all_splits = []
        self.root.all_points(all_splits)
        return sorted(all_splits)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        return 'DT with\nNumber of nodes: {}\nNumber of leaves: {}\nDepth: {}'.\
            format(self.count_node(), self.number_of_leaves, self.get_depth())


class Node:

    target_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

    def __init__(self, col_idx, split_point, chi, frequencies, targets):

        self.split_point = split_point
        '''
        self.split_point = []

        if split_point is None:
            self.split_point = None
        else:
            self.split_point.append(split_point)
        '''
        self.col_idx = col_idx

        self.chi = chi
        self.frequencies = frequencies
        self.targets = targets

        self.children = {}
        # RGB
        rel_freqs = [i / sum(frequencies) for i in frequencies]
        mixed_color = []

        for ctr, f in enumerate(rel_freqs):
            target_color = Node.target_colors[ctr]
            mixed_color.append([f * i for i in target_color])
        color_rgb = np.sum(mixed_color, axis=0)
        color_rgb = [int(round(i)) for i in color_rgb]
        self.hex = '#%02x%02x%02xb0' % tuple(color_rgb)

    def insert(self, X, y, col_idx, split_point, chi, frequencies, targets, is_leaf=False):
        # Continuous node
        if self.split_point is not None:

            child_name = utility.find_interval(X[0, self.col_idx], self.split_point)

            if child_name not in self.children.keys():
                if is_leaf:
                    result = utility.get_most_frequent(y)
                    self.children[child_name] = TerminalNode(frequencies, targets, result)
                else:
                    self.children[child_name] = Node(col_idx=col_idx, split_point=split_point, chi=chi,
                                                     frequencies=frequencies, targets=targets)
                return
            self.children[child_name].insert(X, y, col_idx, split_point, chi,
                                             frequencies, targets, is_leaf)
        # Nominal node
        else:
            children_names = set(X[:, self.col_idx])
            for child_name in children_names:
                if child_name not in self.children.keys():
                    if is_leaf:
                        result = utility.get_most_frequent(y)
                        self.children[child_name] = TerminalNode(frequencies, targets, result)
                    else:
                        self.children[child_name] = Node(col_idx=col_idx, split_point=split_point, chi=chi,
                                                         frequencies=frequencies, targets=targets)
                    return
                self.children[child_name].insert(X, y, col_idx, split_point, chi,
                                                 frequencies, targets, is_leaf)

    def predict(self, x):
        if isinstance(self, TerminalNode):
            return self.result
        # Continuous path
        if isinstance(x[self.col_idx], numbers.Number) and not isinstance(x[self.col_idx], bool):
            child_name = utility.find_interval(x[self.col_idx], self.split_point)
            children_names = self.children.keys()

            main_branch = None

            for name in children_names:
                if child_name in name:
                    main_branch = name

            return Node.predict(self.children[main_branch], x)

        # Nominal path
        else:
            children_names = self.children.keys()
            child_name = None

            for k in children_names:
                if x[self.col_idx] in k:
                    child_name = k

            if child_name is None:
                print('Trying to move on with {}. But cannot find'.format(x[self.col_idx]))

            return Node.predict(self.children[child_name], x)

    def prune(self, limit, alpha):

        # Chi square significance test
        if self is None or isinstance(self, TerminalNode):
            return

        for child in self.children.values():
            child.prune(limit, alpha)

        y_parts = [c.frequencies for c in self.children.values()]

        if not isinstance(self, TerminalNode) and sum(self.frequencies) <= limit:
            # print('Because of limit excess...')
            self.children = {}

        elif not metrics.calculate_chi_square(y_parts=y_parts, targets=self.targets, is_frequency=True, alpha=alpha).__getitem__(1):
            # print('Because of non-significant split...')
            self.children = {}

        if not self.children and not isinstance(self, TerminalNode):
            # print('Found node: \n' + str(self) + '\n-------------------------------------------')
            result = self.targets[np.argmax(self.frequencies)]
            self.__class__ = TerminalNode
            self.result = result

    def merge_results(self, alpha):
        """
        :param alpha: will be used for merge significany
        :return:
        """
        if isinstance(self, TerminalNode):
            return

        for c in self.children.values():
            c.merge_results(alpha=alpha)

        name_instance = list(self.children.keys())[0]

        # if '>=' not in str(name_instance) and '<' not in str(name_instance) and '<=' not in str(name_instance):

        results = []
        terminals = []
        terminal_branches = []

        for i, c in self.children.items():
            if isinstance(c, TerminalNode):
                results.append(c.result)
                terminals.append(c)
                terminal_branches.append(i)

        if len(results) != 0:

            results = list(set(results))

            for res in results:

                temp_names = []
                temp_nodes = []

                for name, c in zip(terminal_branches, terminals):

                    if c.result == res:
                        temp_names.append(name)
                        temp_nodes.append(c)

                total_frequencies = np.zeros(len(self.targets))

                for tn in temp_nodes:
                    total_frequencies += tn.frequencies

                if len(temp_names) > 1:
                    self.children[' or '.join(map(str, temp_names))] = TerminalNode(frequencies=total_frequencies,
                                                                       targets=self.targets, result=res)
                    for n in temp_names:
                        del self.children[n]

            # If there is only one terminal child left, remove the child, make it's parent a terminal node
            if len(self.children) == 1 and isinstance(list(self.children.values())[0], TerminalNode):
                # print('enter removing one child')
                child_name = list(self.children.keys())[0]

                result = self.children[child_name].result

                self.__class__ = TerminalNode
                self.result = result
                self.children = {}

    def count(self):
        counter = 1
        for child in self.children.values():
            counter += child.count()
        return counter

    def count_leaves(self):
        if self is None:
            return 0

        if isinstance(self, TerminalNode):
            return 1

        ret = 0

        for c in self.children.values():
            ret += c.count_leaves()

        return ret

    def depth(self):
        depths = []
        if self is None:
            return 0
        for child in self.children.values():
            depths.append(child.depth())
        if len(depths) > 0:
            return max(depths) + 1
        return 0

    def all_points(self, all_splits):
        if isinstance(self, TerminalNode):
            return

        all_splits.append(self.split_point)

        for c in self.children.values():
            c.all_points(all_splits)

    @staticmethod
    def generate_colors(n):
        r = int(random.random() * 256)
        g = int(random.random() * 256)
        b = int(random.random() * 256)
        step = 256 / n
        for i in range(n):
            r += step
            g += step
            b += step
            r = int(r) % 256
            g = int(g) % 256
            b = int(b) % 256
            Node.target_colors.append((r, g, b))


class TerminalNode(Node):
    def __init__(self, frequencies, targets, result, col_idx=None, split_point=None, chi=None):
        super().__init__(col_idx=col_idx, split_point=split_point, chi=chi, frequencies=frequencies, targets=targets)
        self.result = result
