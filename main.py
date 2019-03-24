import pandas as pd
import utility as util

from tree import DecisionTreeClassifier
from tree_visualization import draw_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


def test():
    df = pd.read_csv('datasets/balanced_bank.csv')

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    col_names = df.columns.values.tolist()

    dictionary = {
                'winter': ['dec', 'jan', 'feb'],
                'spring': ['mar', 'apr', 'may'],
                'summer': ['jun', 'jul', 'aug'],
                'fall': ['sep', 'oct', 'nov']
                 }

    col_names[8] = 'season'
    X[:, 8] = util.group_values(X[:, 8], dictionary)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train, k=20, average=False)

    g = draw_tree(clf, colnames=col_names, target_description={'yes': 'Subscribed', 'no': 'Not Subscribed'},
                  file_name='tree_visualization', view=True)

    predictions = clf.predict(X_test)

    conf = confusion_matrix(y_test, predictions)
    print('\nConfusion Matrix:\n-----------------')
    print(conf)

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


if __name__ == '__main__':
    test()
