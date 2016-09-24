#### Introduction

This project involves investigating a dataset related to the scandal which led to the bankruptcy of the Enron Corporation. Many high-ranking officials within the Houston-based energy company were indicted or convicted of fraud, insider trading, and money laundering. The data here contains a long list of financial and email features of many high-ranking Enron officials. The list contains several "persons of interest" who were the main focus in the fraud case.

#### Task 1: Select what features you'll use.

146 entries makes for a small data set. This makes outliers more dangerous and overfitting harder to avoid.

A single individual's entry:
METTS MARK
salary : 365788
to_messages : 807
deferral_payments : NaN
total_payments : 1061827
exercised_stock_options : NaN
bonus : 600000
restricted_stock : 585062
shared_receipt_with_poi : 702
restricted_stock_deferred : NaN
total_stock_value : 585062
expenses : 94299
loan_advances : NaN
from_messages : 29
other : 1740
from_this_person_to_poi : 1
poi : False
director_fees : NaN
deferred_income : NaN
long_term_incentive : NaN
from_poi_to_this_person : 38

# of NaN's per Feature
poi :  0
salary :  51
deferral_payments :  107
total_payments :  21
loan_advances :  142
bonus :  64
restricted_stock_deferred :  128
deferred_income :  97
total_stock_value :  20
expenses :  51
exercised_stock_options :  44
other :  53
long_term_incentive :  80
restricted_stock :  36
director_fees :  129
to_messages :  60
from_poi_to_this_person :  60
from_messages :  60
from_this_person_to_poi :  60
shared_receipt_with_poi :  60

The sheer number of NaN's is high, but given that we have a mixture of persons of interest and non-persons of interest, it is reasonable.

Features List: ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'bonus_salary_ratio', 'deferred_income_salary_ratio', 'from_poi_ratio', 'to_poi_ratio']

#### Task 2: Remove outliers

A cursory look at the data in the accompanying PDF reveals financial details that are all over the place. We have no reason to believe that these are not legitimate data points and it seems reasonable that they are key to determining the principal targets of investigation in the fraud case. The two non-persons, on the other hand, were removed.

Persons of Interest:  18
Non-POI's:  126
POI %: 12.5

#### Task 3: Create new feature(s) 

I thought of two features to add to the list. The first is bonus to salary ratio and the second is deferred income to salary level. It makes sense that if a figure within Enron received a high bonus or had a lot of deferred income compared to their salary, then they might be receiving benefits for their role in the fraud. I also added two email features: the ratio of emails sent from a person of interest to an individual in question, and the ratio of emails sent from the individual in question to a person of interest.

However, neither of these features were included in the final classifier.

I used SelectKBest to determine the most important features. The precision and recall falls off after 10 features are selected.

Selected Features: ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'shared_receipt_with_poi', 'bonus_salary_ratio']

#### Task 4: Try a variety of classifiers

Starting with a train/test split, I used Naive Bayes, Decision Tree, SVM, Adaboost, Nearest Neighbors, and Random Forest classifiers to determine the most effective ones. 

I also scaled features to a (0, 1) range, but I noticed no major effects after this was implemented.

Precision is the ratio of true positives to the sum of true and false positives, so it is making sure that the classifier is not identifying too many non-POIs as POIs. Recall is the ratio of true positives to the sum of true positives and false negatives, so it makes sure that the classifier is not identifying too many POIs as non-POIs. I used F1-score, or the harmonic mean of precision and recall, to determine the most effective classifiers.

I validated my models using the F1-score, or the harmonic mean of precision and recall, which will help evaluate which classifiers will be most effective when applied to the Enron data.

#### Task 5: Tune your classifier to achieve better than 0.3 precision and recall

Gaussian Naive Bayes, Decision Tree, Nearest Neighbors, and Random Forest classifiers appear to be the most promising at identifying POIs. I tuned these classifiers further by using a stratified shuffle split to determine the best parameters and features in order to boost the performance of the aforementioned classifiers.

Ordinarily a simple train/test split might work, but with the limited amount of data, we'll need to evaluate the model with the same data used to tune it. I tested a range of parameters using GridSearchCV to determine which combination would produce the best results with the Enron data before passing it on to the tester.

The stratified shuffle split in the tester uses 1000 folds, but I used 200 in GridSearchCV here to save time.

Best Estimator: KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='distance'))])
Best Features: ['salary', 'bonus', 'total_stock_value', 'exercised_stock_options']
Best F1 Score: 0.406

#### Results

Tester.py returns the following using the above algorithm:

Accuracy: 0.87800 
Precision: 0.69602	
Recall: 0.36750	
F1: 0.48102	
F2: 0.40581
	
Total predictions: 13000	
True positives:  735	
False positives:  321	
False negatives: 1265	
True negatives: 10679

#### References

Implementing the stratified shuffle split: https://discussions.udacity.com/t/having-trouble-with-gridsearchcv/186377/4
Pipeline help: https://discussions.udacity.com/t/gridsearchcv-not-able-to-find-the-best-configuration/188752/4
SciKit Learn documentation used extensively: http://scikit-learn.org/

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.

-Matt Ignal