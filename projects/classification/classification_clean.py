#%%

from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_roc_curve, plot_confusion_matrix,confusion_matrix, plot_precision_recall_curve, precision_recall_fscore_support, roc_auc_score,auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, KFold


# %%
df = pd.read_csv('./data/data.csv')
# %% we know that we have to predict a categorical variable from only numeric ones.
df.info()

df.head()
#%% the dataset is fully complete and does not require any cleaning, yay!!
df.isna().sum()
# %% each id is unique.
df['id'].apply(str).describe()
#%%
df['id'] = df['id'].apply(str)

#####################
#                   #
# Explore variables #
#                   #
#####################
target = 'diagnosis'
# %%
df[target].value_counts().plot(kind='bar')
#%%
# lets now start with features_mean

# %% removing the target feature and describing the rest
features = df.columns.to_list()
features.pop(features.index(target))
features.pop(features.index('id'))


# %%
len(features)

# %%
for feature in features:
    print(f'***\n{feature}\n***')
    print(df[feature].describe(), '\n')

#%%
####################
#                  #
# data exploration #
#                  #
####################

#%% do some boxplots to explore the data
for feature in features:
    sns.boxplot(data=df, x=target, y=feature)
    plt.figure()
    # plt.clf()
# sns.boxplot(x="day", y="total_bill", data=tips)

#%%
for feature in features:
    sns.distplot(df[df[target] == 'M'][feature], color='r')
    sns.distplot(df[df[target] == 'B'][feature], color='g')
    plt.figure()

#%%

###################
# Start modelling #
###################

# first do feature selection based on forward selection method with 4 models


# now as ou know our diagnosis column is a object type so we can map it to integer value
df[target] = df[target].map({'M': 1, 'B': 0})

models = {
    'logistic_regression': LogisticRegression(),
    'svc': SVC(gamma='auto'),
    'random_forest': RandomForestClassifier(n_estimators=100),
    'gradient_boost': GradientBoostingClassifier()
}

# %%


def evaluate_models(models: dict, df:pd.core.frame.DataFrame = df, target:str = 'diagnosis', feature_list : list = feature_list):
    model_performance = pd.DataFrame({'colnames': []})
    cv = KFold(5)
    scorers = ['accuracy', 'roc_auc', 'precision', 'recall']
    for name, model in models.items():
        relevant_cols = []
        for feature in feature_list:
            relevant_cols.append(feature)
            X = df[relevant_cols]
            y = df[target]

            x_df = pd.DataFrame(cross_validate(model, X, y, cv=cv,
                                            scoring=scorers, return_estimator=False))
            columns_string = ','.join(relevant_cols)
            performance_results = pd.DataFrame(
                {'colnames': [columns_string], 'model': name, 'accuracy': x_df['test_accuracy'].mean(), 'precision': x_df['test_precision'].mean(), 'recall': x_df['test_recall'].mean()})
            # test_roc_auc	test_precision	test_recall

            model_performance = model_performance.append(performance_results)
    
    return model_performance

# %%
model_performance = evaluate_models(models)
model_performance.reset_index(inplace=True, drop=True)
model_performance['length'] = model_performance.colnames.apply(
    lambda x: len(x.split(',')))
model_performance['accuracy'] = model_performance.accuracy.apply(
    lambda x: round(x, 2))
#%%
model_performance.sort_values(
    ['accuracy', 'length'], ascending=[False, True]).head(15)

#%%
model_performance.iloc[81]['colnames'].split(',')
model_performance.iloc[81]
#%%
winning_features = model_performance.iloc[81]['colnames'].split(',')
# fit and evaluate winning model using scores confusion matrix, and roc curve

X = df[winning_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

#%%
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
#%%
rfc.score(X_test, y_test)
#%%
predictions = rfc.predict(X_test)
precision, recall, f_score, support = precision_recall_fscore_support(
    y_test, predictions)
print('model evaluation:\n')
print(f'precision: {precision}\nrecall{recall}\nf-score{f_score}')
#%%
accuracy_score(y_test,predictions)
#%%
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
#%%
tn, fp, fn, tp
#%%
plot_confusion_matrix(rfc, X_test, y_test, cmap='Blues')
#%%
sns.heatmap(confusion_matrix(y_test, predictions), annot= True, cmap = 'Blues')

#%%
plot_precision_recall_curve(rfc, X_test, y_test)
#%%
plot_roc_curve(rfc, X_test, y_test)
#%%
y_score = rfc.predict_proba(X_test)

roc_auc_score(y_test, y_score[:, 1])
#%%

#%%

###########
#
# Feature engineering using pca
#
########

# coming soon ...
pca = PCA(n_components=4)
x = df.loc[:, features].values
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=[
                           'principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])
finalDf = pd.concat([principalDf, df[target]], axis=1)

# %%
finalDf.head()
# %%
finalDf.corr()
