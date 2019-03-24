'''
TEST FILE
import numpy as np
import binning

x = np.array([5,5,5,5,5,5,5,3,3,3,3,3,3,2,2,2,2,2]).reshape(-1, 1)
y = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0])

binner = binning.BinningTree(k=20, average=False).fit(x, y)

import utility

interval = 'x < 3,3 <= x < 7,x >= 7'
points = [3, 7]
utility.merge_intervals(interval_str=interval, points=points)

print(utility.any_missing(np.array([1, 2, 3, '?'], dtype=object), '?'))
'''

import tree
import numpy as np
import pandas as pd
from tree_visualization import draw_tree

df = pd.read_csv('test.csv')
X = df.iloc[:, 0:1].values
y = df.iloc[:, -1].values

col_names = df.columns

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X, y=y, average=False)
g = draw_tree(clf, colnames=col_names, target_description={'positive': 1, 'negative': 0},
              file_name='test_missing_values', view=True)
