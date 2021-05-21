#%%
"""
NOTES:
------
- what should be a structure of the notebook?
- explore variables
- do correlation analysis
- handle outliers
- model fit(LR,SVC,DecTree,Boosted)
- compare models
- refit model after PCA
- compare model performance after PCA (with itself and others)
https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization
"""
#%% import statements
import pandas as pd
import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
# import lightgbm as lgb
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, plot_roc_curve, confusion_matrix, log_loss, plot_precision_recall_curve, precision_recall_fscore_support, classification_report, roc_curve, auc, mean_squared_error, roc_auc_score, plot_roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, cross_val_predict, KFold


# %%
df = pd.read_csv('./data/data.csv')
# %% we know that we have to predict a categorical variable from only numeric ones. 
df.info()
# %%
df.head()
#%% the dataset is fully complete and does not require any cleaning, yay!!
df.isna().sum()
# %% each id is unique.
df['id'].apply(str).describe()
#%%
df['id'] = df['id'].apply(str)
#%%
#####################
#                   #
# Explore variables #
#                   #
#####################
target='diagnosis'
# %%
df[target].value_counts().plot(kind='bar')
#%%
# lets now start with features_mean
# now as ou know our diagnosis column is a object type so we can map it to integer value
df[target] = df[target].map({'M': 1, 'B': 0})
# %% removing the target feature and describing the rest
features = df.columns.to_list()
features.pop(features.index(target))
features.pop(features.index('id'))


# %%
len(features)

# %%
for feature in features:
    print(f'***\n{feature}\n***')
    print(df[feature].describe(),'\n')
# here is some space for a comment describing the necessity to scale the outliers, but first, print quantile values
#%%
#####################
#                   #
# Scale outliers    #
#                   #
#####################
for feature in features:
    print(f'***\n{feature} percentiles:\n***')
    print(df[feature].quantile(
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.98, 0.99, 1]))
# there is a great difference between the 99th and the 100th percentile in texture_mean, perimeter_mean, area_mean,
# however, they are on the same order of magnitude, so I won't deal with them ATM.
#%%

# %% there are still 30 features in the dataset. let's plot a correlation matrix to see if any of these are correlated after which we can reduce the number of features
####################
#                  #
# data exploration #
#                  #
####################

df.corr()
# with the correlation mateix it becomes easier to identify (and remove) those features which are highly correlated with one another
# therefore skewing model performance
# section about perfect predictors (M-F) and multicollinearity (as at the bottom this is a regression model, or at least the baseline is), as well as week predictors
#%% either increase the size or change annotation as this looks ugly AF. 
sns.heatmap(df.corr())
#%% create scatterplot colored by target variable
sns.scatterplot(data=df, x="smoothness_mean", y="texture_mean", hue="diagnosis")

#%% do some boxplots to explore the data
for feature in features:
    sns.boxplot(data=df,x=target, y=feature)
    plt.figure()
    # plt.clf()
# sns.boxplot(x="day", y="total_bill", data=tips)

#%%
for feature in features:
    sns.distplot(df[df[target] == 'M'][feature], color='r')
    sns.distplot(df[df[target] == 'B'][feature], color='g')
    plt.figure()


#%%
######################
#                    #
# Fit coinflip model #
#                    #
######################
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# predict using a coin flip
dummy = DummyClassifier(strategy='uniform',random_state=1)

dummy.fit(X_train, y_train)

dummy.score(X_test,y_test)


#%%
########################
#                      #
# Fit baseline model 2 #
#  logistic regression #
########################

lr1 = LogisticRegression()

lr1.fit(X_train,y_train)

lr1.score(X_test,y_test)

# so from this just alone, the logistic regression does an accuracy score of 92%. 

#%%

cv = KFold(10)
scorers = ['accuracy', 'roc_auc', 'precision', 'recall']
pd.DataFrame(cross_validate(lr1, X, y, cv=cv, scoring=scorers, return_estimator=True))
#%%
#####################
#                   #
# Feature selection #
#                   #
#####################

# add forward selection procedure



# manual selection from the correlation matrix
relevant_cols1 = ['radius_mean', 'perimeter_mean', 'area_mean',
                  'radius_worst', 'concave points_mean', 'concave points_worst', 'perimeter_worst']
# so just by playing around and selecting columns with rho > 0.7, i got accuracy score = 0.9736842105263158
X = df[relevant_cols1]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)
lr2 = LogisticRegression()

lr2.fit(X_train, y_train)

lr2.score(X_test, y_test)
#%%

y_pred = lr2.predict(X_test)
#%%
confusion_matrix(y_test,y_pred)

#%%

plot_precision_recall_curve(lr2,X_test,y_test)
#%%
#########################
#                       #
# Define compare models #
#                       #
#########################

X = df[features]
y = df[target]

models = {
    'logistic_regression': LogisticRegression(),
    'svc': SVC(gamma='auto'),
    'random_forest': RandomForestClassifier(n_estimators=100)
}
def compare_classification_models(models, X, y, no_iterations):
    model_scores = pd.DataFrame(columns=[0, 1, 2])
    for i in range(no_iterations):
        model_scores = model_scores.append(pd.DataFrame.from_dict(
            {0: ['***'], 1: [f'iteration {i+1}'], 2: ['***'], 3: ['***'], 4: ['***']}))
        for name, model in models.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2)
            print(f'fitting model: {name} on iteration {i}')
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            precision, recall, f_score, support = precision_recall_fscore_support(y_test,predictions)
            accuracy = accuracy_score(y_test,predictions)
            df1 = pd.DataFrame(
                {name: [precision, recall, f_score, support,accuracy]}).transpose()
            model_scores = model_scores.append(df1)
    model_scores.rename(
        columns={0: 'precision', 1: 'recall', 2: 'f_score',3:'support',4:'accuracy'}, inplace=True)
    return model_scores


# %%
#####################################################
#                                                   #
# compare and evaluate models based on performance  #
#                                                   #
#####################################################


a = compare_classification_models(models, X, y, 5)
a
# explain what a confusion matrix is. also explain on which metrics you want to improve on.
#%%
############################################
#
# Run same models on updated feature set
#
#######################################


X = df[relevant_cols1]
y = df[target]

models = {
    'logistic_regression': LogisticRegression(),
    'svc': SVC(gamma='auto'),
    'random_forest': RandomForestClassifier(n_estimators=100)
}


def compare_classification_models(models, X, y, no_iterations):
    model_scores = pd.DataFrame(columns=[0, 1, 2])
    for i in range(no_iterations):
        model_scores = model_scores.append(pd.DataFrame.from_dict(
            {0: ['***'], 1: [f'iteration {i+1}'], 2: ['***'], 3: ['***'], 4: ['***']}))
        for name, model in models.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2)
            print(f'fitting model: {name} on iteration {i}')
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            precision, recall, f_score, support = precision_recall_fscore_support(
                y_test, predictions)
            accuracy = accuracy_score(y_test, predictions)
            df1 = pd.DataFrame(
                {name: [precision, recall, f_score, support, accuracy]}).transpose()
            model_scores = model_scores.append(df1)
    model_scores.rename(
        columns={0: 'precision', 1: 'recall', 2: 'f_score', 3: 'support', 4: 'accuracy'}, inplace=True)
    return model_scores


b = compare_classification_models(models, X, y, 5)
b

#%%
#######
#     #
# PCA #
#     #
#######
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
x = df.loc[:, features].values
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])
finalDf = pd.concat([principalDf, df[target]], axis=1)

# %%
finalDf.head()
# %%
finalDf.corr()

# sns.heatmap(correlation, annot=True, cmap='vlag', annot_kws={"fontsize": 22})

# %%
# Train and compare winning models with Original Data and PCA data

#%%
X = finalDf[['principal component 1', 'principal component 2',
             'principal component 3', 'principal component 4']]
y = finalDf[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)
lr3 = LogisticRegression()

lr3.fit(X_train, y_train)

lr3.score(X_test, y_test)
#%%

y_pred = lr3.predict(X_test)
#%%

#####
#
# confusion matrix
#
#####
confusion_matrix(y_test, y_pred)
#%%
y_prob = lr3.predict_proba(X_test)

# %%

#
# correlation 
##
correldf = df.corr()
# %%
correldf.iloc[0,:].sort_values(ascending = False)[1:]
# %%
feature_list = correldf.iloc[0, :].sort_values(ascending=False)[1:].index
# %%
relevant_cols2 = []
model_performance = pd.DataFrame({'colnames':[]})
print(model_performance)
#%%
cv = KFold(5)
scorers = ['accuracy', 'roc_auc', 'precision', 'recall']
lr2 = LogisticRegression()

for feature in feature_list:
    # fit 5 times, select min, max avg and append to dict with feature name
    relevant_cols2.append(feature)
    X = df[relevant_cols2]
    y = df[target]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2)
    x_df = pd.DataFrame(cross_validate(lr2, X, y, cv=cv,
                                scoring=scorers, return_estimator=False))
    columns_string = ','.join(relevant_cols2)
    performance_results = pd.DataFrame({'colnames':[columns_string], 'accuracy':x_df['test_accuracy'].mean()})
    # print(performance_results)
    model_performance = model_performance.append(performance_results)
    # lr2.fit(X_train, y_train)

    # model_performance[','.join(relevant_cols2)] = (lr2.score(X_test, y_test))

    # y_pred = lr2.predict(X_test)
   
    # confusion_matrix(y_test, y_pred)

    # plot_precision_recall_curve(lr2, X_test, y_test)

# %%
model_performance.reset_index(inplace = True, drop = True)
model_performance.sort_values('accuracy', ascending=False)
# %%
rfc = RandomForestClassifier(n_estimators=100)
relevant_cols2 = []
model_performance = pd.DataFrame({'colnames': []})

cv = KFold(5)
scorers = ['accuracy', 'roc_auc', 'precision', 'recall']

for feature in feature_list:
    # fit 5 times, select min, max avg and append to dict with feature name
    relevant_cols2.append(feature)
    X = df[relevant_cols2]
    y = df[target]

    x_df = pd.DataFrame(cross_validate(rfc, X, y, cv=cv,
                                       scoring=scorers, return_estimator=False))
    columns_string = ','.join(relevant_cols2)
    performance_results = pd.DataFrame(
        {'colnames': [columns_string], 'accuracy': x_df['test_accuracy'].mean()})

    model_performance = model_performance.append(performance_results)

# %%
model_performance.reset_index(inplace=True, drop=True)
model_performance.sort_values('accuracy', ascending=False)

# %%
model_performance.iloc[21].colnames
# %%

###################
#
# how to compare multiple models using forward feature selection
#
############
models = {
    'logistic_regression': LogisticRegression(),
    'svc': SVC(gamma='auto'),
    'random_forest': RandomForestClassifier(n_estimators=100),
    'gradient_boost': GradientBoostingClassifier()
}

# %%
model_performance = pd.DataFrame({'colnames': []})
cv = KFold(5)
scorers = ['accuracy', 'roc_auc', 'precision', 'recall']
for name, model in models.items():
    relevant_cols2 = []
    for feature in feature_list:
        relevant_cols2.append(feature)
        X = df[relevant_cols2]
        y = df[target]

        x_df = pd.DataFrame(cross_validate(model, X, y, cv=cv,
                                        scoring=scorers, return_estimator=False))
        columns_string = ','.join(relevant_cols2)
        performance_results = pd.DataFrame(
            {'colnames': [columns_string],'model':name, 'accuracy': x_df['test_accuracy'].mean(), 'precision': x_df['test_precision'].mean(),'recall':x_df['test_recall'].mean()})
        # test_roc_auc	test_precision	test_recall

        model_performance = model_performance.append(performance_results)

# %%
model_performance.reset_index(inplace=True, drop=True)
model_performance.sort_values('accuracy', ascending=False).head(10)

# %%
model_performance.head(10)
# %%
win_cols = model_performance.iloc[85].colnames.split(',')
# %%
len(win_cols)
# %%
len(df.columns)
# %%
# which is the cheapest model with the best performance? i.e. contains the least cols
model_performance['length'] = model_performance.colnames.apply(lambda x: len(x.split(',')))
model_performance['accuracy'] = model_performance.accuracy.apply(lambda x: round(x,2))
# %%
model_performance.sort_values([ 'accuracy','length'], ascending = [False,True]).head(15)
# %%
model_performance[model_performance.model == 'logistic_regression'].sort_values('accuracy', ascending=False)
# %%
