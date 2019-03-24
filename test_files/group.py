'''
TEST FILE
Created on Aug 31, 2018

@author: mertkelkit
'''

import utility
import numpy as np

arr = np.array(['june', 'july', 'july', 'april', 'january', 'april', 'may', 'april', 'december'])
dictionary = {'winter': ['december', 'january', 'february'],
              'spring': ['march', 'april', 'may'],
              'summer': ['june', 'july', 'august'],
              'fall': ['september', 'october', 'november']}

arr = utility.group_values(arr, dictionary)
print(arr)


from split import Splitter

X = np.array([[5,2,3,1], [4, 7, 2, 1], [5, 3, 2, 5], [8, 3, 1, 2], [8, 8, 2, 1]])
y = np.array([True, False, False, False, True])

sp = Splitter().fit(X, y)
x_parts, y_parts = sp.multiway_split(1, [3, 7])
print(x_parts)
print(y_parts)