
# coding: utf-8

# # Identifying Fraud in Enron emails and financial data
# ### by Andrii Zakharov
# 
# In this project, I will try to answer the question whether it is possible to identify Persons of Interest (POIs) in the Enron fraud case by looking at a dataset of Enron employees' emails and their financial data. Even with the data's comparably small size, however, I would not be able to infer much from it by way of manual analysis. Machine learning to the rescue! 
# 
# This dataset is a perfect case for supervised learning classification algorithms, as it contains POI identifiers, and so can be used to both train classifiers and assess their performance. Let's get to it!

# In[1]:

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[2]:

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# After loading the data, I will convert it to a data frame for easier inspection and feature manipulation.

# In[3]:

# create a data frame from dictionary and inspect it
df = pd.DataFrame.from_dict(data_dict, orient="index")
print df.info()
print "###############################################"
print "We have", df.shape[0], "observations of", df.shape[1], "features,", df.shape[0]*df.shape[1], "total data points"
nan_count = 0
for feat in df.columns.values:
    if 'NaN' in list(df[feat]):
        nan_count += df[feat].value_counts()['NaN']
print nan_count, "of those are NaNs, or", round((float(nan_count) / (df.shape[0]*df.shape[1])) *100, 1), "%"
print "###############################################"
print "Number of POIs:", sum(df["poi"]), "out of", len(df["poi"])
print "It's", round(df["poi"].mean(), 3) * 100, "%"


# This is the basic structure of the data. We see feature names, shape, number of NaN values and POIs in our data. Quite a few NaN's there! And only very few POIs, so very imbalanced target classes.
# 
# One thing that occurs to me is that I can drop the feature "email_address", as it will have a unique value for each observation and so won't be able to contribute much to the analysis.
# 
# Another thing that I remember from the lessons is that the dataset actually contains a row with sum totals for the financial data. Those would definitely be outliers, so I'll drop that row as well.

# In[4]:

df.drop('email_address', axis=1, inplace=True)
df.drop("TOTAL", axis=0, inplace=True)


# We probably have more outliers in the data, but I would be cautious about simply removing them, as some may contain valuable information about our targets. I will z-transform each feature in the data, define outliers as +-3 SD, and see how many POIs will be found among the outliers. 

# In[5]:

# define outlier searching function
from scipy.stats import zscore
def is_outlier(feat, thresh=3):
    feat_z = zscore(feat)
    return abs(feat_z) > thresh


# In[6]:

# identify outliers and compare POIs
poi = np.array(df["poi"])
for feat in df.columns.values:
    f = np.array(df[feat])
    f[f=="NaN"] = 0
    f[f=="None"] = 0
    f = f.astype(float)
    poi_out = poi[is_outlier(f)]
    print "n outliers in " + feat, len(poi_out)
    print "Prop poi's:", poi_out.mean().round(3)


# We see that there are not very many outliers in general, and some of the more frequent ones (e.g. total_stock_value and bonus) contain a much more extreme proportion of POIs than data average (0.8 and 0.75 vs 0.12).
# 
# Based on that, I'm reluctant to take any radical steps here, so I'll let the ourliers be for now.
# 
# What I will do, however, is add a couple new features to the data. 
# 
# First, I noticed there are people who got really huge boni (bonuses?) from Enron, over $1 million. That seems suspicious to me, so I'll include this as an extra discreet feature (> or < 1 mil.) into the dataset. 
# 
# Second, as we have around 44% missing values, and some persons have much more missing information about them than others, I wanted to include the proportion of missing data per person as a feature. Maybe those POIs were more secretive somehow.
# 
# So I'll include these two new features and compose a new feature list.

# In[7]:

# add feature 1: discreet 1/0 for bonus over 1 mil
huge_bonus = []
for bonus in df["bonus"]:
    if bonus > 1000000 and bonus != "NaN":
        huge_bonus.append(1)
    else:
        huge_bonus.append(0)
df["huge_bonus"] = huge_bonus


# In[8]:

# add feature 2: proportion NaNs for given person
prop_nans = []
rows = df.iterrows()
for p in range(len(df)):
    person = rows.next()[1]
    prop_nans.append(len(person[person=="NaN"]) / float(len(person)))
df["prop_nans"] = prop_nans


# In[9]:

features_list = list(df.columns.values)
features_list.remove("poi")
features_list = ["poi"] + features_list
features_list


# Let's format the data with these features and run a quick classifier comparison with all the popular classifiers, using the test function from tester.py. Just to get a first impression of how they perform here, before doing any feature selection. Note that we don't care about feature scaling here as I'm not using any clustering, which would be sensitive to that. 
# 
# Also note that the test function uses Stratified Shuffle Split for cross-validation. Cross-validation is important as it allows us to realistically assess our model's performance. If we would use our whole data set for both fitting our classifier and predicting the target classes, we would inevitably overestimate the model's performance - our model would overfit, i.e. adjust itself too well to the data we have. Then, when we'd try to predict labels for a new dataset, chances are the model wouldn't do too well. To avoid this problem, one approach would be to split our data into a training set and a test set, and use only the training set for model fitting, and only the test set for prediction. Cross-validation automates this concept. In case of the Stratified Shuffle Split, it divides the data into a specified number of subsets (called folds), shuffles their order, and then fits the model to the dataset comprised of all the folds but one, which is used for testing. This process is repeated with each fold serving as a testing fold once. At each step, model performance is assessed and at the end an average is produced.
# 
# It is particularly important to use cross-validation in our case, as we have quite imbalanced target classes. This means that just one train/test split would probably not have the same proportion of labels in both the training and testing data, which would be a problem for model assessment. The Stratified Shuffle Split is a particularly good method to deal with imbalanced classes and hence it is used here.

# In[10]:

my_dict = df.to_dict(orient="index")

data = featureFormat(my_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[11]:

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from tester import test_classifier

# run quick classifier comparison
clf = GaussianNB()
test_classifier(clf, my_dict, features_list)
print "######################################"

clf = DecisionTreeClassifier()
test_classifier(clf, my_dict, features_list)
print "######################################"

clf = RandomForestClassifier()
test_classifier(clf, my_dict, features_list)
print "######################################"

clf = AdaBoostClassifier()
test_classifier(clf, my_dict, features_list)
print "######################################"


# Oh nice, it looks like AdaBoost is actually not far off the mark for our goal of minimum 0.3 for both precision and recall. It actually has ~0.37 precision, but not enough recall at 0.28. Random Forest has comparable precision, but much worse recall, and Naive Bayes actually does surprisingly well on recall, but lacks in precision.
# 
# All in all, AdaBoost still looks most promising to me. Maybe a grid search will be able to tune it up further?
# 
# The idea behind the grid search is simple: instead of manually testing different parameter combinations for the model, a function will do this automatically. The inputs are the ranges of parameters to try, and the function goes over all possible combinations of parameters within these ranges (i.e. searching the grid). The performance is assessed using cross-validation and measured by a specified scoring function. In the end, the classifier with optimized parameters for the specific metric is obtained.
# 
# Since the tester uses Stratified Shuffle Split, I'll use it as well for cross-validation in my grid search, with 100 splits. It's crucial to use cross-validation for the reasons I described earlier.  I'll also use "f1" for scoring, as we're interested in both precision and recall, and search over a small range of two main AdaBoost parameters.

# In[ ]:




# In[12]:

# run grid search on AdaBoost
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=100, random_state=42)
params = {'n_estimators': range(48, 52),
          'learning_rate': np.arange(1., 2., 0.1)}

ada_clf = GridSearchCV(AdaBoostClassifier(random_state=42), params, cv=sss, scoring="f1")
ada_clf.fit(features, labels)
print "Best f1 score:", ada_clf.best_score_
print ada_clf.best_estimator_


# In[13]:

# test the best AdaBoost classifier
test_classifier(ada_clf.best_estimator_, my_dict, features_list)


# ...and funny enough, the tuned-up AdaBoost performs worse than before the grid search! With 49 estimators and a learning rate of 1.4 it is only able to produce 0.33 precision and 0.24 recall with the tester function. 
# 
# We'll have to do better than that! Let's try some feature selection and see if it helps.
# 
# First of all, let's see if the two features I added are actually helping. I'll run the standard AdaBoost without them and see if the performance drops.

# In[14]:

# test without my engineered features
features_list_orig = list(features_list)
features_list_orig.remove("huge_bonus")
features_list_orig.remove("prop_nans")

clf = AdaBoostClassifier()
test_classifier(clf, my_dict, features_list_orig)
print "######################################"


# Wow, the performance actually went up! We're at ~0.4 precision and 0.3 recall. Those were not very helpful features it seems... But now I'm curious whether there are more features in the way here. This warrants a more systematic approach.
# 
# I'll now implement 10 decision trees and use their aggregate feature importances to decide which features to keep. I'll start with the full set of features and test different thresholds of importance, comparing the resulting feature sets' performance. 

# In[15]:

data = featureFormat(my_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# use DT feature importances for feature selection
feat_imps = np.zeros(len(features_list)-1)
clf_dt = DecisionTreeClassifier(random_state=42)

for _ in range(10):
    clf_dt.fit(features, labels)
    feat_imps +=  clf_dt.feature_importances_

feat_imps = feat_imps.tolist()
feat_names_imps = {}
features_list.remove("poi")
features_list.sort()

for i in range(len(features_list)):
    feat_names_imps[features_list[i]] = round(feat_imps[i], 2)

feat_names_imps


# I'll compare three crude thresholds: above 0, above 0.5 and above 1.0 feature importance.

# In[16]:

features_list0 = ["poi"]
features_list05 = ["poi"]
features_list1 = ["poi"]
for key in feat_names_imps:
    if feat_names_imps[key] > 0:
        features_list0.append(key)
    if feat_names_imps[key] >= 0.5:
        features_list05.append(key)
    if feat_names_imps[key] >= 1:
        features_list1.append(key)
print features_list0
print features_list05
print features_list1


# In[23]:

clf = AdaBoostClassifier()
test_classifier(clf, my_dict, features_list0)
print "######################################"
clf = AdaBoostClassifier()
test_classifier(clf, my_dict, features_list05)
print "######################################"
clf = AdaBoostClassifier()
test_classifier(clf, my_dict, features_list1)
print "######################################"


# Here are the comparison results for the three thresholds with the standard AdaBoost:
# 
# Feature importance threshold | Precision | Recall
# -----------------------------|-----------|-------
# 0.0                          |0.39382    |0.29950
# 0.5                          |0.42244    |0.30500
# 1.0                          |0.46987    |0.40550
# 
# It's clear that we should use the smallest set of features with over 1.0 feature importance.
# This set only has 5 predictors, 3 from the financial data and 2 from the email data.
# 
# I'll now run a quick comparison of the other three classifiers with this shortened set of features, just out of interest.

# In[26]:

# run quick classifier comparison on new data
clf1 = GaussianNB()
test_classifier(clf1, my_dict, features_list1)
print "######################################"

clf1 = DecisionTreeClassifier()
test_classifier(clf1, my_dict, features_list1)
print "######################################"

clf1 = RandomForestClassifier()
test_classifier(clf1, my_dict, features_list1)
print "######################################"


# Ok, so here are the results. Our AdaBoost now sports ~0.47 precision and ~0.41 recall. The simple Decision Tree also got boosted with the same 0.41 recall, but a lower ~0.36 precision. Random forest has good precision, and Naive Bayes shows outstanding recall, but they both lack on the other metrics. Note that this performance impovement was achieved by using only the 5 best features of the 21 total. The joys of feature selection! The power of sparsity!
# 
# Now I'll try the grid search for AdaBoost one more time, with the same parameter grid, just to see if any improvement happens this time.

# In[19]:

data = featureFormat(my_dict, features_list1, sort_keys = True)
labels, features = targetFeatureSplit(data)

# run second grid search on AdaBoost on new data with same params
ada_clf.fit(features, labels)
print "Best f1 score:", ada_clf.best_score_
print ada_clf.best_estimator_


# In[20]:

# test the best AdaBoost classifier once again
test_classifier(ada_clf.best_estimator_, my_dict, features_list1)


# Nope, it got worse again. The grid search-produced best classifier with 49 estimators and 1.7 learning rate has around 0.03 worse precision and recall than the standard 50 estimators, 1.0 learning rate AdaBoost.
# 
# One idea would be to try and tweak the grid search further with a custom scoring function. I have an impression that the f1-scoring still isn't optimal for our case. One could also try to search over a much larger parameter space.
# 
# However, I feel that these nuances lie outside the scope of this project. I pronounce the standard AdaBoost the winner! After all, it achieved an impressive ~0.47 precision and ~0.41 recall in the tester function with Stratified Shuffle Split. 
# 
# In human-understandable language it means that the AdaBoostClassifier is able to identify 41% of real POIs, and 47% of persons it classifies as POIs are actually ones. And it only uses 5 features to achieve that: 'expenses', 'exercised_stock_options', 'from_messages', 'shared_receipt_with_poi', and 'director_fees'.
# 
# 
# ## Impressive!
# (not really, but it'll do)

# In[25]:

# export my classifier, data, and features
my_dataset = my_dict
features_list = features_list1

dump_classifier_and_data(clf, my_dataset, features_list)

