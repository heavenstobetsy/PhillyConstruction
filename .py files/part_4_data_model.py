#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import boruta
from boruta import BorutaPy


# ## Part 4: Data modeling

# ### Part 4.a Add in a target:

# In[50]:


# Create a target variable is where there are 1+ unsafe building notices

permit_inspect_unsafe['target'] = 0
permit_inspect_unsafe.loc[permit_inspect_unsafe['unsafe_building']>0, 'target'] = 1

#Data clean up
permit_inspect_unsafe=permit_inspect_unsafe[permit_inspect_unsafe['permitnumber']!='461321`']
permit_inspect_unsafe[permit_inspect_unsafe['unsafe_building']==1].head(1)
print('This address has too many unsafe flags and is skewing the dataset:', permit_inspect_unsafe[permit_inspect_unsafe['addresskey']==461088].addresskey.count())

#Cleanup
permit_inspect_unsafe=permit_inspect_unsafe[permit_inspect_unsafe['addresskey']!=461088]


# In[51]:


#Data cleanup

permit_inspect_unsafe.drop([ 'APPL','SITE','addresskey','permitnumber','unsafe_building'], axis=1, inplace=True)
permit_inspect_unsafe['target'] = permit_inspect_unsafe['target'].apply(np.float64)

permit_inspect_unsafe = permit_inspect_unsafe.loc[:,~permit_inspect_unsafe.columns.duplicated()]

permit_inspect_unsafe.fillna(0, inplace=True)


# In[52]:


#Very skewed data:
print("original dataset's distribution:")
print ('shape: ', permit_inspect_unsafe.shape)
print(permit_inspect_unsafe.target.value_counts(normalize=True))


# ### Part 4.b Resample and reshape the data, as it is very skewed. Resampling: Undersample or Oversample?
# 

# ### Part 4.b.1 Undersampling:

# In[53]:


#Resample data:

permit_inspect_unsafe_0 = permit_inspect_unsafe.loc[permit_inspect_unsafe.target==0]
permit_inspect_unsafe_1 = permit_inspect_unsafe.loc[permit_inspect_unsafe.target==1]

# rebalance the training set so it has 50-50% for the target. 
# this has a random state, so it needs to be fixed to be able to reproduce results. 
ix = sample_without_replacement(n_population=permit_inspect_unsafe_0.shape[0],
                                n_samples=permit_inspect_unsafe_1.shape[0], 
                                random_state=42)

permit_inspect_unsafe_0 = permit_inspect_unsafe_0.iloc[ix]
permit_inspect_unsafe_undersample = pd.concat([permit_inspect_unsafe_0, permit_inspect_unsafe_1], ignore_index=True)

#shuffle dataframe so train/test split wont' get messed up 
permit_inspect_unsafe_undersample = shuffle(permit_inspect_unsafe_undersample)

permit_inspect_unsafe_undersample.reset_index(inplace=True, drop=True)

print ('\n')
print ('rebalanced statistisc: ')
print ('shape: ', permit_inspect_unsafe_undersample.shape)
permit_inspect_unsafe_undersample.target.value_counts(normalize=True)


# ##### Initial accuracy, recall, and confusion matrix:

# In[54]:


# Separate input features and target
y = permit_inspect_unsafe_undersample.target
X = permit_inspect_unsafe_undersample.drop('target', axis=1)

# setting up testing and training sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(x_train, y_train)



#Validation Results
print('Validation Results')
print("Accuracy score test", clf_rf.score(x_val, y_val))
print("Recall score test", recall_score(y_val, clf_rf.predict(x_val)))

print ("roc_auc_score train", roc_auc_score(y_train, clf_rf.predict_proba(x_train)[:,1]))
print ("roc_auc_score test", roc_auc_score(y_val, clf_rf.predict_proba(x_val)[:,1]))

#Validation Results 2 (Just cause)
# make predictions for test data
y_pred = clf_rf.predict(x_val)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_val, predictions)
print("Accuracy score test: %.2f%%" % (accuracy * 100.0))


# In[55]:


from sklearn.metrics import confusion_matrix
test_conf_matrix=confusion_matrix(y_val, y_pred)
test_conf_matrix = pd.DataFrame(test_conf_matrix)

test_conf_matrix.index=['true_negative', 'true_positive']
test_conf_matrix.columns=['pred_negative', 'pred_positive']
test_conf_matrix


# ##### Visualizing the ROC curve:

# In[56]:


# plot probability distribution

prediction = clf_rf.predict_proba(X=permit_inspect_unsafe_undersample.iloc[:,:-1])
prediction = pd.DataFrame(prediction).iloc[:,1]

permit_inspect_unsafe_undersample_t = permit_inspect_unsafe_undersample.copy()
permit_inspect_unsafe_undersample_t.reset_index(drop=True, inplace=True)
permit_inspect_unsafe_undersample_t['prediction'] = prediction


# In[57]:


# visualize the ROC curve

fpr, tpr, _ = metrics.roc_curve(permit_inspect_unsafe_undersample_t.target, permit_inspect_unsafe_undersample_t.prediction)
auc = metrics.roc_auc_score(permit_inspect_unsafe_undersample_t.target, permit_inspect_unsafe_undersample_t.prediction)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.grid()
plt.show()


# ### Part 4.b.2 Oversampling using SMOTE:

# In[58]:


from imblearn.over_sampling import SMOTE

# Separate input features and target
y = permit_inspect_unsafe.target
X = permit_inspect_unsafe.drop('target', axis=1)

# setting up testing and training sets
training_features, test_features,training_target, test_target, = train_test_split(X,y,
                                               test_size = .25,random_state=42)


# By oversampling only on the training data, none of the information in the validation data is being used to create synthetic observations. So these results should be generalizable. 

# In[59]:


#Oversampling on only the training data:
x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .25,
                                                  random_state=42)


# In[60]:


#SMOTE
sm = SMOTE(random_state=42, sampling_strategy ='auto')
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

print('shape:',x_train_res.shape)


# In[61]:


#Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(x_train_res, y_train_res)


# In[62]:


print('Validation Results')
print("Accuracy:",clf_rf.score(x_val, y_val))
print ("Recall:",recall_score(y_val, clf_rf.predict(x_val)))
print( '\nTest Results')
print ("Accuracy:",clf_rf.score(test_features, test_target))
print ("Recall:",recall_score(test_target, clf_rf.predict(test_features)))

The validation results closely match the unseen test data results, which is exactly what I would want to see after putting a model into production.

However, while the precision is high, recall is poor using SMOTE. Why does SMOTE perform so poorly?  This paper suggests that SMOTE performs poorly when given high-dimensional class-imbalanced data.
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-106
# ### Part 4.b Conclusion: Based on the above results, I've chosen to undersample rather than use SMOTE

# #### Interesting sidenote--charts showing how SMOTE resamples data:

# In[63]:


# Oversample and plot imbalanced dataset with SMOTE
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where

# define dataset
features = [f for f in permit_inspect_unsafe.columns if f not in ['target']]
X = permit_inspect_unsafe[features].values
y = permit_inspect_unsafe['target'].values.ravel()

# summarize class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()


# In[64]:


# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()


# ### Part 4.c Feature Selection & Clean Up:
Pruning features with Boruta: Boruta can help reduce overfitting the model. When building a machine learning model, it can be hard it is to identify which features are important and which are just noise. Removing the noisy features will help with memory, computational cost and the accuracy of the model.  Also, by removing features you will help avoid the overfitting of the model.##### Having too many irrelevant features in your data can decrease the accuracy of the models. Three benefits of performing feature selection before modeling the data are:

1. Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
2. Improves Accuracy: Less misleading data means modeling accuracy improves.
3. Reduces Training Time: Less data means that algorithms train faster.

##### https://mbq.github.io/Boruta/
# In[65]:


#Random Forest model for boruta:

# rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
rf = RandomForestClassifier(random_state=42)
random_state=42


# In[66]:


#Boruta feature selector with max iterations=150
import boruta
from boruta import BorutaPy

features = [f for f in permit_inspect_unsafe_undersample.columns if f not in ['target']]
len(features)

X = permit_inspect_unsafe_undersample[features].values
y = permit_inspect_unsafe_undersample['target'].values.ravel()

boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, max_iter = 150, perc = 90)
boruta_feature_selector.fit(X, y)


# In[67]:


#New number of selected features

print ('\n Number of selected features: {}'.format(boruta_feature_selector.n_features_))


# In[68]:


#Filter on the selected features

X_filtered = boruta_feature_selector.transform(X)
print ('\n New shape: {}'.format(X_filtered.shape))


# In[69]:


# Features that have been pruned:

final_features = list()
indexes = np.where(boruta_feature_selector.support_ == False)
for x in np.nditer(indexes):
    final_features.append(features[x])
print("Features that have been pruned: {}".format(final_features))


# In[70]:


#Create a list of the selected feature names if we would like to use them at a later stage.

final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(features[x])
print("Features that have been kept: {}".format(final_features))


# #### New dataframe with pruned features:

# In[71]:


df2=(permit_inspect_unsafe_undersample.reindex(columns=['censustract', 'code_violations_count', 'Failed', 'Closed', 'Passed', 'unique_insp_cases', 'CONSTRUCTION SERVICES', 'HAZARD', 'NON HAZARDOUS', 'UNSAFE', 'CLOSEDCASE', 'COMPLIED', 'ABATE', 'CLOSE', 'CMPLY', 'REISS', 'RES', 'SR', 'STP', 'CONTRACTOR', 'OWNER', 'ALTERATION PERMIT','target']))
df2.head(1)


# ### Part 4.d Separate Models & Ensemble Learning:
# 

# In[72]:


# split data into X and y
X= df2.iloc[:,:-1]
y=df2.iloc[:,-1]

#Or another way:
# Separate input features and target
y = df2.target
X = df2.drop('target', axis=1)

# setting up testing and training sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)


# ### Part 4.d.1 Separate Models:
# 

# In[73]:


# A host of Scikit-learn models
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

def get_models():
    """Generate a library of base learners."""
 #   nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
   # lr = LogisticRegression(max_iter=5000,random_state=42)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=42,max_iter=1000)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    rf1 = RandomForestClassifier(n_estimators=10, max_features=3, random_state=42)
    rf2 = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier()

    models = {
              'svm': svc,
              'knn': knn,
             # 'naive bayes': nb,
              'mlp-nn': nn,
              'random forest 1': rf1,
              'random forest 2': rf2,
              'gbm': gb,
             # 'logistic': lr,
              'xgb': xgb
              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((y_val.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(x_train, y_train)
        P.iloc[:, i] = m.predict_proba(x_val)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models (ROC-AUC):")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))

    print("Done.\n")


####################################### 
        
def accuracy_score_models(model_list):
    """Score model in prediction DF"""
    print("Scoring models (Accuracy):")
    for i, (name, m) in enumerate(models.items()):
        m.fit(x_train, y_train)
        y_pred = m.predict(x_val)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_val, predictions)
        print('%s: Accuracy %.3f' % (name, accuracy))
        
    
def evaluate_models(model_list):
    print("Scoring models (RMSE):")
    for i, (name, m) in enumerate(models.items()):
        m.fit(x_train, y_train)
        yhat = m.predict(X)
        mse = mean_squared_error(y, yhat)
        print('%s: RMSE %.3f' % (name, sqrt(mse)))  
        
        
def evaluate_models123(P, y):
    print("Scoring models (RMSE):")
    for m in P.columns:
            yhat = m.predict(X)
            mse = mean_squared_error(y, yhat)
            #print('%s: RMSE %.3f' % (name, sqrt(mse)))    
            tmp_res = pd.DataFrame([name, sqrt(mse)]).T
            tmp_res.columns=['model_name', 'mse']
            result = pd.concat([tmp_res, tmp_res])
    return result
        #tmp_res.head()
        

AUC ROC indicates how well the probabilities from the positive classes are separated from the negative classes:
# In[74]:


models = get_models()
P = train_predict(models)
score_models(P, y_val)


# In[75]:


models = get_models()

# evaluate accuracy score
accuracy_score_models(models)


# In[76]:


from math import sqrt

# evaluate RMSE for base models
evaluate_models(models)


# In[77]:


#Best model (the random forest 2) summary:

# Separate input features and target
y = df2.target
X = df2.drop('target', axis=1)

# setting up testing and training sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

        
        
rf2 = RandomForestClassifier(random_state=42)
rf2.fit(x_train, y_train)

y_pred = rf2.predict(x_val)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_val, predictions)
        
test_conf_matrix=confusion_matrix(y_val, y_pred)
test_conf_matrix = pd.DataFrame(test_conf_matrix)

test_conf_matrix.index=['true_negative', 'true_positive']
test_conf_matrix.columns=['pred_negative', 'pred_positive']
test_conf_matrix

# auc_roc=metrics.classification_report(y_val,y_pred)
# auc_roc

For an ensemble strategy to work, prediction errors must be relatively uncorrelated.
# ### Part 4.d.2 Ensemble Model:
# 

# In[78]:


#ML-Ensemble
#Errors are significantly correlated, which is to be expected for models that perform well, 
#since it’s typically the outliers that are hard to get right. However, the model correlations below fall in 
#the 50-80% range, so there is room for model improvement through ensemble.

from mlens.visualization import corrmat

corrmat(P.corr(), inflate=False)
plt.show()


# In[79]:


#Looking at error correlations on a class prediction basis:
#Scores at lower, which is more promising

corrmat(P.apply(lambda pred: 1*(pred >= 0.5) - y_val.values).corr(), inflate=False)
plt.show()


# In[80]:


print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(y_val, P.mean(axis=1)))


# In[81]:


#Looking at ROC curve for all models

from sklearn.metrics import roc_curve

def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    cm = [plt.cm.rainbow(i)
      for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]

    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(y_val, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    fpr, tpr, _ = roc_curve(y_val, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.show()


plot_roc_curve(y_val, P.values, P.mean(axis=1), list(P.columns), "ensemble")


# In[82]:


#Ranking of models by 
# A simple check shows that some models perform worse than others.

p = P.apply(lambda x: 1*(x >= 0.5).value_counts(normalize=True))
p.index = ["UNSAFE", "OKAY"]
p.loc["OKAY", :].sort_values().plot(kind="bar")
#plt.axhline(0.25, color="k", linewidth=0.5)
#plt.text(0., 0.23, "True share")
plt.show()

A small improvement for ROC-AUC score:
# In[83]:


#Improve the Emsemble by removing the worst offender: Naive Bayes

include = [c for c in P.columns if c not in ["svm"]]
print("Truncated ensemble ROC-AUC score: %.3f" % roc_auc_score(y_val, P.loc[:, include].mean(axis=1)))


# ### Part 4.d.2 Ensemble Model:

# In[84]:


#Define a meta learner

meta_learner = RandomForestClassifier(
#     n_estimators=1000,
#     loss="exponential",
#     max_features=4,
#     max_depth=3,
#     subsample=0.5,
#     learning_rate=0.005,
    random_state=42)


# In[85]:


# example of a super learner using the mlens library
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlens.ensemble import SuperLearner

# create a list of base-models
def get_models():
    models = list()
    models.append(GaussianNB())
    models.append(SVC(C=100, probability=True))
    models.append(KNeighborsClassifier(n_neighbors=5))
    models.append(LogisticRegression(max_iter=5000,random_state=42))
    models.append(MLPClassifier((80, 10), early_stopping=False, random_state=42,max_iter=1000))
    models.append(GradientBoostingClassifier(random_state=42))
    models.append(RandomForestClassifier(max_features=3, random_state=42))
    models.append(RandomForestClassifier(random_state=42))
    models.append(XGBClassifier())
    
#     models = {
#               'svm': svc,
#               'knn': knn,
#               'naive bayes': nb,
#               'mlp-nn': nn,
#               'random forest 1': rf1,
#               'random forest 2': rf2,
#               'gbm': gb,
#               'logistic': lr,
#               'xgb': xgb
#               }

    return models

# create the super learner
def get_super_learner(X):
    ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X))
    # add base models
    models = get_models()
    ensemble.add(models)
    # add the meta model
    ensemble.add_meta(LogisticRegression(solver='lbfgs'))
    return ensemble


# In[86]:


# create the inputs and outputs
y = df2.target.values
X = df2.drop('target', axis=1).values

# # split
# X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)

# setting up testing and training sets
X, x_val, y, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
print('Train', X.shape, y.shape, 'Test', x_val.shape, y_val.shape)

# create the super learner
ensemble = get_super_learner(X)


# fit the super learner
ensemble.fit(X, y)
# # summarize base learners
print(ensemble.data)
# # make predictions on hold out set
yhat = ensemble.predict(x_val)
print('Super Learner: %.3f' % (accuracy_score(y_val, yhat) * 100))

The super learner ensemble model scores slightly lower (-1%) than the second random forest model, which had an accuracy rate of %85.9.  I'll continue to tweak the super learner model to improve the accuracy and ROC-AUC score.