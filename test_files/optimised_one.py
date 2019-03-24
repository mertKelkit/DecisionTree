# MAIN TEST FILE


import pandas as pd
import utility as util

from tree import DecisionTreeClassifier
from tree_visualization import draw_tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

'''
df = pd.read_csv('breast_cancer_train.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
col_names = df.columns.values.tolist()


clf = DecisionTreeClassifier()
clf = clf.fit(X, y, average=False)
print('\n\n\n\n\n\n\n')

df = pd.read_csv('breast_cancer_test.csv')
x_test = df.iloc[:, :-1].values
y_test = df.iloc[:, -1].values
print('Test data size {}x{}'.format(df.shape.__getitem__(0), df.shape.__getitem__(1)))
g = draw_tree(clf, colnames=col_names, target_description={2: 'Benign', 4: 'Malignant'}, file_name='test', view=True)

predictions = clf.predict(x_test)
my_conf = confusion_matrix(y_test, predictions)
print('My tree\'s confusion matrix:')
print(my_conf)

from sklearn.tree import DecisionTreeClassifier


skclf = DecisionTreeClassifier(criterion='entropy', splitter='best').fit(X, y)

pred = skclf.predict(x_test)
print('Scikit-learn\'s tree\'s confusion matrix:')
print(confusion_matrix(y_test, pred))



from timeit import default_timer as timer

df = pd.read_csv('tennis.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
col_names = df.columns.values.tolist()

# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=421, test_size=0.33)

print(df)
clf = DecisionTreeClassifier()
clf = clf.fit(X, y, k=2)


g = draw_tree(clf, colnames=col_names, file_name='testtennis', view=True)

'''
df = pd.read_csv('balanced_bank.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

col_names = df.columns.values.tolist()

dictionary = {'winter': ['dec', 'jan', 'feb'],
              'spring': ['mar', 'apr', 'may'],
              'summer': ['jun', 'jul', 'aug'],
              'fall': ['sep', 'oct', 'nov']
             }

col_names[8] = 'season'
X[:, 8] = util.group_values(X[:, 8], dictionary)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9840)


clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train, k=20, average=False)

g = draw_tree(clf, colnames=col_names, target_description={'yes': 'Subscribed', 'no': 'Not Subscribed'},
              file_name='banking_last_wednesday_experimental', view=True)

predictions = clf.predict(x_test)

conf = confusion_matrix(y_test, predictions)
print('\nConfusion Matrix:\n-----------------')
print(conf)


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


f1 = f1_score(y_test, predictions, average='macro')
pre = precision_score(y_test, predictions, average='macro')
re = recall_score(y_test, predictions, average='macro')
acc = accuracy_score(y_test, predictions)

print('------------------------------')
print('My prediction scores:')
print('------------------------------')
print('F1 score is: %f' % f1)
print('Precision score is: %f' % pre)
print('Recall score is: %f' % re)
print('Accuracy score is: %f' % acc)

'''
df = pd.read_csv('flight_delays_train.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

col_names = df.columns.values.tolist()

clf = DecisionTreeClassifier()
clf = clf.fit(X, y, k=500, average=False)

g = draw_tree(clf, colnames=col_names, target_description={'N': 'No delay', 'Y': 'Delay'},
              file_name='big_data_flights', view=True)
'''