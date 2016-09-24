#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
# Import classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Import metrics
from sklearn.metrics import f1_score, classification_report
# Import validation methods
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Amount of entries:", len(data_dict.keys())
print

# Individual entry
name = data_dict.keys()[0]
print name
for elem in data_dict[name]:
    if elem in features_list:
        print elem, ":", data_dict[name][elem]
print

# Remove non-persons
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

poi = sum([1 for k,v in data_dict.items() if v["poi"]])
not_poi = sum([1 for k,v in data_dict.items() if not v["poi"]])
print "Persons of Interest: ", poi
print "Non-POIs: ", not_poi
print "POI %:", poi / float(poi + not_poi)*100
print

print "# of NaN's per Feature"
for i in features_list:
    print i, ": ", sum([1 for k,v in data_dict.items() if v[i] == "NaN"])
print

# Create new features
# Remove NaNs and 0s where appropriate
for key in data_dict:
    if (data_dict[key]['salary'] != 'NaN' and
                data_dict[key]['salary'] != 0 and
                data_dict[key]['bonus'] != 'NaN' and
                data_dict[key]['bonus'] != 0):
        data_dict[key]['bonus_salary_ratio'] =\
            float(data_dict[key]['bonus']) / \
            float(data_dict[key]['salary'])
    else:
        data_dict[key]['bonus_salary_ratio'] = 'NaN'

for key in data_dict:
    if (data_dict[key]['salary'] != 'NaN' and
                data_dict[key]['salary'] != 0 and
                data_dict[key]['deferred_income'] != 'NaN'):
        data_dict[key]['deferred_income_salary_ratio'] =\
            float(data_dict[key]['deferred_income']) / \
            float(data_dict[key]['salary'])
    else:
        data_dict[key]['deferred_income_salary_ratio'] = 'NaN'

for key in data_dict:
    if (data_dict[key]['from_poi_to_this_person'] != 'NaN' and
                data_dict[key]['from_poi_to_this_person'] != 0 and
                data_dict[key]['from_this_person_to_poi'] != 'NaN' and
                data_dict[key]['from_this_person_to_poi'] != 0):
        data_dict[key]['from_poi_ratio'] = \
            float(data_dict[key]['from_poi_to_this_person']) / \
            float(data_dict[key]['from_this_person_to_poi'])
        data_dict[key]['to_poi_ratio'] = \
            float(data_dict[key]['from_this_person_to_poi']) / \
            float(data_dict[key]['from_poi_to_this_person'])
    else:
        data_dict[key]['from_poi_ratio'] = 'NaN'
        data_dict[key]['to_poi_ratio'] = 'NaN'

my_dataset = data_dict

features_list.append('bonus_salary_ratio')
features_list.append('deferred_income_salary_ratio')
features_list.append('from_poi_ratio')
features_list.append('to_poi_ratio')
print "Features List:", features_list
print

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# The precision and recall falls off after 10 features are selected.
kbest = SelectKBest(k=10)
kbest.fit_transform(features, labels)

# Determine Features
important = [features_list[i + 1] for i in kbest.get_support(indices=True)]

# Add 'poi'
features_list = ['poi'] + important
print "Selected Features:", features_list
print

# Repeat feature selection
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

target_names = ['Non-POI', 'POI']

# Set up classifier functions

def naive_bayes(features_train, features_test, labels_train, labels_test):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Classifier: Gaussian Naive Bayes"
    print "F1 Score:  %.3f" % f1_score(pred, labels_test)
    print classification_report(labels_test, pred, target_names=target_names)

def tree(features_train, features_test, labels_train, labels_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Classifier: Decision Tree"
    print "F1 Score:  %.3f" % f1_score(pred, labels_test)
    print classification_report(labels_test, pred, target_names=target_names)

def svm(features_train, features_test, labels_train, labels_test):
    clf = SVC(kernel="rbf")
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Classifier: Support Vector Machine"
    print "F1 Score:  %.3f" % f1_score(pred, labels_test)
    print classification_report(labels_test, pred, target_names=target_names)

def AdaBoost(features_train, features_test, labels_train, labels_test):
    clf = AdaBoostClassifier(random_state=42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Classifier: Adaboost"
    print "F1 Score:  %.3f" % f1_score(pred, labels_test)
    print classification_report(labels_test, pred, target_names=target_names)

def NN(features_train, features_test, labels_train, labels_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Classifier: Nearest Neighbors"
    print "F1 Score:  %.3f" % f1_score(pred, labels_test)
    print classification_report(labels_test, pred, target_names=target_names)

def RandomForest(features_train, feature_test, labels_train, labels_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Classifier: Random Forest"
    print "F1 Score:  %.3f" % f1_score(pred, labels_test)
    print classification_report(labels_test, pred, target_names=target_names)

# Start with train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Feature Scaling - No major effects after this was implemented
scaler = MinMaxScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# Call the classifiers
naive_bayes(features_train, features_test, labels_train, labels_test)
tree(features_train, features_test, labels_train, labels_test)
svm(features_train, features_test, labels_train, labels_test)
AdaBoost(features_train, features_test, labels_train, labels_test)
NN(features_train, features_test, labels_train, labels_test)
RandomForest(features_train, features_test, labels_train, labels_test)

print "#########################################################"

# Best performing classifiers
gnb = GaussianNB()
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
nn = KNeighborsClassifier()

# GridSearchCV is used to compare parameters. Parameters were tuned manually
# based on their performance using an increasing amount of folds. The
# following represents the final iteration of tuning.
params_gnb = {'kbest__score_func': [f_classif],
              'kbest__k': [5, 6, 7, 8]}

params_rfc = {}

params_dtc = {}

params_nn = {'nn__weights': ['uniform', 'distance'],
             'nn__n_neighbors': [4, 5, 6],
             'nn__leaf_size': [1, 2],
             'nn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
             'kbest__score_func': [f_classif],
             'kbest__k': [3, 4]}

parameters = [params_gnb, params_rfc, params_dtc, params_nn]

clf_list = [('gnb', gnb), ('rfc', rfc), ('dtc', dtc), ('nn', nn)]

# The stratified shuffle split in the tester uses 1000 folds, but we'll use
# 200 here to save time.
for clf_i in range(len(clf_list)):
    cv = StratifiedShuffleSplit(labels, n_iter=200, random_state=42)
    kbest2 = SelectKBest(f_classif, k='all')
    pipeline = Pipeline([('kbest', kbest2), clf_list[clf_i]])
    gs = GridSearchCV(pipeline, scoring='f1', cv=cv,
                      param_grid=parameters[clf_i])
    gs.fit(features, labels)
    pred = gs.predict(features)
    print "Best Estimator:", gs.best_estimator_
    features_selected_bool = gs.best_estimator_.named_steps['kbest']\
        .get_support()
    features_selected_list = [x for x, y in zip(features_list[1:],
                                                features_selected_bool) if y]
    print "Best Features:", features_selected_list
    print "Best F1 Score: %.3f" % gs.best_score_
    print

# Best Classifier according to F1 score is KNearestNeighbors.
clf = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='distance')
features_list = ['poi', 'salary', 'bonus', 'total_stock_value',
                 'exercised_stock_options']

# Dump your classifier, dataset, and features_list.
dump_classifier_and_data(clf, my_dataset, features_list)