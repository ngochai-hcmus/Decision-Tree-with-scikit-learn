import graphviz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix

def prepare_dataset():
  index=[]
  board = ['a','b','c','d','e','f','g']
  for i in board:
      for j in range(6):
          index.append(i + str(j+1))

  column_names  = index + ['Class']

  url = 'https://raw.githubusercontent.com/ngochai-hcmus/Decision-Tree-with-scikit-learn/main/connect-4.data'

  df = pd.read_csv(url, names = column_names)

  le = preprocessing.LabelEncoder()
  for col in df.columns:
    df[col] = le.fit_transform(df[col])

  feature = df.drop(['Class'], axis=1)
  label = df['Class']

  return feature, label


def decision_tree_classifiers(feature, label):
  test = [60, 40, 20,  10]

  for proportion in test:
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=proportion/100, random_state=42)
    #Building the decision tree classifiers
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    clf = clf.fit(feature_train, label_train)
    dot_data = export_graphviz(clf, out_file=None, feature_names=feature.columns, class_names=['draw', 'loss', 'win'],rounded=True, filled=True)
    graph = graphviz.Source(dot_data)
    graph.render('output/tree_{}_{}'.format(100-proportion, proportion), cleanup=True)

    #Evaluating the decision tree classifiers
    predictions = clf.predict(feature_test)
    accuracy = accuracy_score(label_test, predictions)
    print("Train/test = ", f"{100-proportion}/{proportion}:")
    print("Accuracy: ", accuracy)
    print(classification_report(label_test, predictions, target_names=['draw','loss', 'win'], zero_division=0))

    plot_confusion_matrix(clf, feature_test, label_test)
    filename = 'output/confusion_matrix_{}_{}'.format(100-proportion, proportion)
    plt.savefig(filename)
    plt.clf()
  return

def decision_tree_80_20(feature, label):
  feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.2, random_state=42)
  depths = [None, 2, 3, 4, 5, 6, 7]
  for depth in depths:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    clf = clf.fit(feature_train, label_train)
    dot_data = export_graphviz(clf, out_file=None, feature_names=feature.columns, class_names=['draw', 'loss', 'win'],rounded=True, filled=True)
    graph = graphviz.Source(dot_data)
    graph.render('output/decision_tree_80_20_depth_{}'.format(depth), cleanup=True)

    #Evaluating the decision tree classifiers
    predictions = clf.predict(feature_test)
    accuracy = accuracy_score(label_test, predictions)
    print("- Decision tree 80/20 with depth:", f"{depth}")
    print("Accuracy: ", accuracy)
    
  return


feature, label = prepare_dataset()
decision_tree_classifiers(feature, label)
decision_tree_80_20(feature, label)
