# 2.1 loading dataset

## Import necessary libraries
import pandas as pd

# will be used for anova analysis
from pingouin import anova
# will be used for scaling sata
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

path = "./dataset.csv"
data = pd.read_csv(path)

# 2.2 data preparation

print("Data Preparation")
print("-------------------------------------------")
# Observe data
print(data.head())

# There is an unneeded column "Unnamed:0" that appeared while creating read_csv, let's move that
data.drop("Unnamed: 0", axis=1, inplace=True)

# let's check information about the dataframe
print("Dataframe informations")
print(data.info())

"""
From this metadata information, we can find that we have 5000 total index, float64, int64, bool
and object types for columns. By looking data we see that this object column (corresponding to feature_1)
contains object values (certainly dates). We also observe that some features (feature_20, feature_26,
feature_27, feature_28, feature_29, feature_30) miss some data but the number of missing data in each of them
is low.
"""

# Now let's check statistics of numerical values (min, max, variance...)
print("\n Dataframe description")
print(data.describe())

# From this description we see that the standard deviations of our data and target is really high and
# the range also is high. We will do standardisation to have more regular data

# We see that for the feature_21 has only one unique value (min, max, mean are equals and standard deviation equals 0)
# This feature is not useful for us as at it does'nt play a role on the target, we remove it.
data.drop("feature_21", inplace=True, axis=1)

# Before continuing let's standardise numerical values
scaler = StandardScaler()
#select the numerical_columns by removing bool and object column names
numerical_columns = list(data.select_dtypes(exclude=["object", "bool"]).columns)

# we remove the bool columns because we don't want it to be standardized as we will loose
# their True/False signification (0/1)
data[numerical_columns]= scaler.fit_transform(data[numerical_columns])

# Now we add booleans as they can be used as numerical
boolean_columns = list(data.select_dtypes(include=["bool"]).columns)
numerical_columns.extend(boolean_columns)

# Now as said before there are missing values we need to deal with.
# There are many strategies for dealing with it but here we prefer to use
# KNN to fill the missing values as it will allow us to have different
# and certainly more accurate values. Using a strategy which consists of
# replacing with the mean/mode/constant can affect negatively our data (even as we have few missing values
# and that KNN is more expensive).

# we choose 20 so that the model only consider really close values
imputer = KNNImputer(n_neighbors=20)
data[numerical_columns]= imputer.fit_transform(data[numerical_columns])

# Okay now let's check the importance of features tu see if we can remove some of them or reduce dimension

# Firstly we check the importance of the different numerical values
# To do so, we observe the correlation of the features with the target.
cor = data.corr()
#We put the plot in comments for the file to run without interruption but you can remove to visualize

"""
plt.figure(figsize=(15,10))
sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
plt.show()
"""

print("\n Correlations with target")
print(cor["target"])

# we see that there are columns that are not relevant for the target as their correlation coefficient
# is really close to zero.
# Let's keep a track on the relevant columns (we consider a column as relevant if the absolute value
# of the correlation with the target is >0.5)
cor_target = abs(cor["target"])
relevant_columns=list(cor_target[cor_target > 0.5].index)
relevant_columns.remove("target")
print("\n The number of columns with an absolute correlation > 0.5 is: ".format(len(relevant_columns)))

# Now there are maybe high correlation between those columns, as it's not necessary to have two high correlated
# columns let's remove the high correlated ones (those with correlation>=0.9)
high_corr_rate = 0.9
removed_columns = set()
for i in range(len(relevant_columns)):
    for j in range(i+1, len(relevant_columns)):
        if cor[relevant_columns[i]][relevant_columns[j]] >= high_corr_rate:
            removed_columns.add(relevant_columns[j])

for col in removed_columns:
    relevant_columns.remove(col)

print("\n The number of finally selected columns is: {}".format(len(relevant_columns)))
# we see that we only have 6 numerical columns that are really useful (It can be interesting for models like
# LinearRegression for example) but are those features sufficiant to produce a really good model after?
# (The answer will be given from the PCA analysis)

# we can now remove target from numerical variables
numerical_columns.remove("target")

# Let's try the PCA to see how many components are necessary to retrieve a high part of the target variance.
pca = PCA()
data_pca = pca.fit_transform(data[numerical_columns])

# Let's plot the cumulative explained variance ratio
# (We have commented it so that when the file is run it finishes without interruption, but you can move comments
# to observe

cumsum = pca.explained_variance_ratio_.cumsum()
"""
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(0,1)
plt.style.context('seaborn-whitegrid')

plt.plot(cumsum)
plt.show()
"""

print("\n Array of cumulative explained variance ratio")
print(cumsum)

# we define here a threshold of 95% for selecting components.
threshold = 0.95
n_components = 0
for i in range(len(cumsum)):
    if cumsum[i]> threshold:
        n_components = i+1
        break

print("\n The {} first components allow us to have {} % of the total explained variance".format(n_components,threshold*100))

# We see that with the 11 first components we can retrieve almost all the variance of the data (95%)

# Having done PCA and feature selection, we observe that for the PCA we are using 11 features and for feature selection 6.
# But let's also remark that the 6 first components of PCA allow us to have only 83.5% of the total explained variance.
# So our 6 selected features cannot explain more than that.

# We prefer continuing with the PCA's data because they also allow us to reduce highly the initial dataframe and helps
# us to keep the variance

# create a dataframe for those components
data_pca = pd.DataFrame(data=data_pca[:, 0:n_components], columns=["component_{}".format(i+1) for i in range(n_components)])

# We have now finished with the numerical values,

# Finally let's check the importance of the object column
# Now as feature_1 has an object type let's see if it is relevant using One-way ANOVA.
# So we try to see if the feature_1 column has an importance on the target column
# more precisely if variations of this feature affect the target.
aov = anova(data=data, dv="target", between="feature_1", detailed=True)
print("\n One way Analysis of Variance")
print(aov)

# From this ANOVA we observe that the p-value is really low, that means we cannot remove this categorical
# feature. It is important.

# Now let's see how we will consider the object column (which contains certainly dates)
# Let's see how many distinct values there are.
print("\n Number of object values: {}".format(len(data["feature_1"].unique())))

"""
We observe that we only have 8 values for this feature (which represents certainly dates).
As those are dates we have 2 principal choices, firstly transform those dates in categories and
secondly divide this column in 3 columns for day, month and year.

For the first method (transform in categories), the advantage is that we only have 8 possible values
in our dataset so this representation can be really meaningful for our model. The problem is that we
don't know what those dates are representing. Maybe in the production model we will have new dates and
just plotting them as unknown can be harmful.

For the second method, we can create 3 columns corresponding for the day, month and year.
The advantage is that we will be able to take into account new dates. But there is a big inconvenient
our model only has few dates and those dates are not even complete. Transforming like this will result
in creating too much missing data and will be harmful for our model.

Conclusion: Finally we decide to transform the dates in categories. As we don't know the role of this feature
it's better to use it as categories and when possible ask more information to the manager.
"""

# transform that column to 8 new columns corresponding to the categorical columns
data = pd.get_dummies(data, columns=["feature_1"])

# create arrays of column names of categorical features and numerical features
columns = data.columns
categorical_columns = [col for col in columns if "feature_1_" in col]
numerical_columns = [col for col in columns if col not in categorical_columns]
numerical_columns.remove("target")

# let's add those categorical_columns in our data_fs and data_pca dataframes
data_pca[categorical_columns]=data[categorical_columns]

# now we can split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(data_pca, data["target"],
                                                                    test_size=0.2, shuffle=True, random_state=51)

# 2.3 model training

print("\nModel training")
print("-------------------------------------------\n")


# useful imports
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# there are many interesting models for regression: Linear Regression, Polynomial regression, Ridge regression
# Lasso Regression, ElasticNet Regression, XGBoost, LGBRegressor, Support Vector Machine (SVM), Neural Networks...

# We are going to consider for this task three models, Polynomial Regression (including linear model), XGBoost, SVM.
# Firstly Polynomial Regression, because it can quickly be introduced as baseline, XGBoost and SVM because they have shown really good results
# since their creation


# The strategy will be to use cross validation for each of those models (to avoid overfitting) and the score we try to maximize is the
# correlation R2 between our prediction and the target.

# number of folds for validation
folds = 5

# Pipeline for Polynomial Regression
print("\n Polynomial Regression with Grid Search\n")

poly_pipeline = Pipeline([('poly_features', PolynomialFeatures()), ('model', LinearRegression())])

# degrees of the tested models
params = {"poly_features__degree": [1,2,3]}

# try different polynomial models
poly_grid = GridSearchCV(poly_pipeline, cv=folds, scoring='r2', param_grid=params)
# fit the model and do cross validation
poly_grid.fit(X_train,y_train)

print("Global results")
print(poly_grid.cv_results_)

print("\nBest score for polynomial model: {}".format(poly_grid.best_score_))
print(poly_grid.best_params_)

# We see that the best cross validation score for polynomial model is -1.46 obtained with
# the linear model (degree=1). This is really bad and means that the model performs poorly on
# one or more validation sets (when R2 score is lower than 0 (which is not possible in theory
# but possible in Python according to the documentation), that means that the model performs
#really poorly and the R2 value for that set makes that the average R2 for all folds is bad as we can see)
# while looking in depth we see that the majority of folds have a R2 around 0.72 but one of the folds has -9.46
# this is why the global score is so bad.

# when we look other degrees (2 and 3) we see that the model performs really poorly on validation sets
# it means it overfits on training sets. Those polynomial models are not adapted

# Let's just keep the baseline obtained with Linear Regression and compute XGBoost model

# XGBoost Model

print("\n XGBoost Model with Random Search \n")

# We will also do random search to find good parameters

# We define the list of parameters we want to test.
# We vary the depth of the tree, the number of estimators, the learning rate (eta)
# the booster is chosen between work with a set of trees or a set of linear functions
# the gamma and alpha helps for L1 and L2 regularizations to be sure the model will
# not overfit

params = {
        'min_child_weight': [1, 5],
        'eta': [0.3, 0.02],
        'booster': ['gbtree', 'gblinear'],
        'n_estimators': [100, 200, 400, 500, 700, 1000]
        }

# the estimator
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', max_depth=8, gamma=1, alpha=1)

# the number of candidates
# this value is selected because of the computation constraint.
param_comb = 5

# we will do a random_search where we will be trying to maximize the r2 score of the predictions and target
# without overfitting
random_search_xgb = RandomizedSearchCV(xg_reg, param_distributions=params, n_iter=param_comb,
    scoring='r2', n_jobs=-1, cv=folds, verbose=3,
                                   random_state=1001 )

random_search_xgb.fit(X_train, y_train)

print('\n Best estimator:')
print(random_search_xgb.best_estimator_)
print('\n Best score:')
print(random_search_xgb.best_score_)

# the best validation R2 is 0.72

# Now let's compute the SVM Regression model

# SVM Regression Model
svm = SVR()
params = {
    # the model kernel will be linear or Radial Basis Function
    'kernel': ['rbf', 'linear'],
    # regularization parameter
    'C': np.arange(1, 11, 2),
    # importance accorded to one training example
    'gamma': ['auto', 'scale', 0.2]
}

# we will do a random_search where we will be trying to maximize the r2 score of the predictions and target
# without overfitting
random_search_svm = RandomizedSearchCV(svm, param_distributions=params, n_iter=param_comb,
    scoring='r2', n_jobs=-1, cv=folds, verbose=3,
                                   random_state=1001 )

random_search_svm.fit(X_train, y_train)

print('\n Best estimator:')
print(random_search_svm.best_estimator_)
print('\n Best score:')
print(random_search_svm.best_score_)

# Interesting the SVM validation R2 is 0.786 which is higher than the validation of the
# other models

# 2.4 model evaluation (evaluate model perf and display metrics)
print("\nModel evaluation")
print("-------------------------------------------\n")

# linear model
y_predict_lm = poly_grid.predict(X_test)

# xgboost with randomsearch
y_predict_xgb = random_search_xgb.predict(X_test)
# svm with randomsearch
y_predict_svm = random_search_svm.predict(X_test)

# store r2_scores
r2_lm = r2_score(y_predict_lm, y_test)
r2_xgb = r2_score(y_predict_xgb, y_test)
r2_svm = r2_score(y_predict_svm, y_test)

# store root mean squared errors
rmse_lm = mean_squared_error(y_predict_lm, y_test)
rmse_xgb = mean_squared_error(y_predict_xgb, y_test)
rmse_svm = mean_squared_error(y_predict_svm, y_test)

model_names = ["linear model", "xgboost", "SVM"]
metrics = [[r2_lm, r2_xgb, r2_svm], [rmse_lm, rmse_xgb, rmse_svm]]

result_df = pd.DataFrame(data=metrics, columns=model_names, index=["r2_score", "rmse"])

print("\nEvaluation metrics for the different models")
print(result_df)

# We can see from this that only the SVM model is performing well.
# It has a R2 correlation of 0.804 and a RMSE really low (0.1)

# We also observe that linear model is performing better than xgboost for this test data

# Now let's plot prediction error

fig,a =  plt.subplots(3,1)

# store results for prediction error
results = [y_predict_lm, y_predict_xgb, y_predict_svm]

for j in range(3):
    a[j].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, c='#000000')
    a[j].scatter(y_test, results[j])
    a[j].set_title('Prediction error of {}'.format(model_names[j]))
    a[j].set_ylabel('Predicted')

plt.xlabel('Measured')
plt.show()

# We observe, as metrics show before, that only SVM model predicts data that are really close to the real data.
# This is the final model we will use.

# But it is also important to observe that all the models had to struggle with the same data (we can see that
# there are some points far away from the y=x curve.

"""
At the end of our study, we maintain the SVM model SVR(C=7, gamma=0.2).
It is important to note that to adapt easily our model to new data we will need to use Pipelines
with the different steps we followed (filling of data, pca, categorical values and svm).
"""
# Thanks !
