import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import time

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from titanic.titanic_functions import setup_environment
from common.preprocessing import identify_categorical, create_categorical_transformer, create_feature_preprocessor
from common.preprocessing import create_numerical_transformer, get_transformer_feature_names
from common.visualization import plot_feature_histogram, plot_feature_importance_xgb


#  PROJECT DEFINITION
# Setup project variables and paths
start_time = time.time()
PLOT_SHOW = False
SAVE_DATA = True
PROJECT_NAME = "titanic"

data_path, output_path = setup_environment(PROJECT_NAME)

# Load up data
print("Loading data ...")
gender_submission = pd.read_csv(os.path.join(data_path, "gender_submission.csv"))
submission_data = pd.read_csv(os.path.join(data_path, "test.csv"))
train = pd.read_csv(os.path.join(data_path, "train.csv"))

# Define key labels
class_label = "Survived"
target_key = "PassengerId"
print("Submission cols:", sorted(submission_data.columns))
print("Train cols:", sorted(train.columns))
print(train.info())
print(train.head())

# DATA PREPROCESSING
print("\nPreprocessing data ...")

# Clean up existing columns
train["Cabin_Sector"] = train["Cabin"].astype(str).str[0]
submission_data["Cabin_Sector"] = submission_data["Cabin"].astype(str).str[0]
print(sorted(train["Cabin_Sector"].unique()))
print(submission_data["Cabin_Sector"].unique())

train["Cabin_Locale"] = train["Cabin"].astype(str).str[1:]
submission_data["Cabin_Locale"] = submission_data["Cabin"].astype(str).str[1:]

# Identify and characterize categorical variables
s = (train.dtypes == 'object')
categorical_cols = identify_categorical(train)
numerical_cols = list(train.columns.values)
numerical_cols = list(np.setdiff1d(numerical_cols, categorical_cols))
print("Numeric columns:", numerical_cols)

# Handle Missing Data (Impute)
print("\nHandling missing data...")
# Number of missing values in each column of training data
missing_val_count_by_column = (train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Create Preprocessing Transformers
categorical_transformer = create_categorical_transformer(strategy="most_frequent")  #most_frequent
numerical_transformer = create_numerical_transformer(strategy="constant", fill_value=None)  #strategy constant, none

# Feature generation
#TODO exponential combination
#TODO automated feature generation
#TODO Explore differences between train data and submission data

# FEATURE REPORTING
# Print Data
print("\nTRAINING")
print(train.describe())

print("\nSUBMISSION")
print(submission_data.describe())

women = train.loc[train.Sex == 'female'][class_label]
rate_women = sum(women)/len(women)
print("\n% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male'][class_label]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)
print(len(train[train[class_label] == 1]), len(train[train[class_label] == 0]))

# Plot data
sns.set(style="ticks", color_codes=True)
g_train = sns.pairplot(train, diag_kind="hist", hue="Survived")
plt.tight_layout()
SAVE_DATA and plt.savefig(os.path.join(output_path, f'{PROJECT_NAME}_features_pairplot.png'), bbox_inches='tight')
PLOT_SHOW and plt.show()
print()

# plot histograms
plot_feature_histogram(train, numerical_cols, class_col=class_label, hist_check_int=True)
SAVE_DATA and plt.savefig(os.path.join(output_path, f'{PROJECT_NAME}_features_histograms.png'), bbox_inches='tight')
PLOT_SHOW and plt.show()

# TRAIN MODEL PIPELINE
# Feature Selection
# select_features_num = ["Parch", "Pclass", "SibSp",]
# select_features_cat = ["Sex", "Embarked",]
select_features_num = ["Parch", "Pclass", "SibSp", ]  #Age #"Fare",
select_features_cat = ["Sex", "Embarked",]# "Cabin_Sector",]
select_features = select_features_num + select_features_cat
print("Selected Features: ", select_features)

# Define Model Training Inputs
y = train[class_label]
X = train[select_features]

# Preprocessing
scaler = StandardScaler()
# pca = PCA()
preprocessor = create_feature_preprocessor(numerical_transformer, select_features_num, categorical_transformer,
                                           select_features_cat)

# Train model
# TODO try semi supervised learning

# Setting random state forces the classifier to produce the same result in each run
n_cv = 5  # cv=5 is default
scorer = "accuracy"

# model = RandomForestClassifier(random_state=random_state)
model = xgb.XGBClassifier()
# model = lgb.LGBMClassifier()

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', scaler),
    # ('pca', pca),
    ('model', model)
])

param_grid = {
    'preprocessor__num__strategy': ['mean', 'median', 'most_frequent', 'constant'],
    'preprocessor__cat__imputer__strategy': ['constant', 'most_frequent'],

    # 'pca__n_components': [5, 15, 30, 45, 64],
    'model__n_estimators': [10, 50, 75, 100, 125, 200],

    # usually max_depth is 6,7,8
    'model__max_depth': list(range(2, 10)),

    # learning rate is around 0.05, but small changes may make big diff
    'model__learning_rate': [0.03, 0.05, 0.07, 0.09, 0.1],
    # 'model__subsample':  list(map(lambda x: x * 0.1, range(1, 10))),
    # "model__early_stopping_rounds": 10,
    # "model__verbose": False,

    # 'model__colsample_bytree': list(map(lambda x: x * 0.1, range(1, 15))),
    # 'model__min_child_weight': list(range(1, 15)),
    # tuning min_child_weight subsample colsample_bytree can fight against overfit
    # 'model__objective': list(range(2, 15)),
    # gamma, alpha, lambda  #finally, ensemble xgboost with multiple seeds may reduce variance
}

# brute force scan for all parameters, here are the tricks
# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['binary:logistic'],
#               'silent': [1],
#               'missing':[-999],
#               'seed': [1337]}

fit_params = {
                # "xgbrg__eval_set": [(val_X, val_y)],
                "xgbrg__early_stopping_rounds": 10,
                "xgbrg__verbose": False}

print("\nPerforming GridSearch on pipeline")
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=n_cv, scoring=scorer, return_train_score=True, refit=True,
                      verbose=1)
search.fit(X, y)  #, fit_params=fit_params)
best_model = search.best_estimator_

print("\nSelected Features: ", select_features)
print("Best parameter (%s score=%0.3f):" % (search.scorer_, search.best_score_))
print(search.best_params_)
print(best_model.named_steps["model"])

print("mean_train_score: ", search.cv_results_["mean_train_score"].mean(), search.cv_results_["mean_train_score"].std())
print("std_train_score: ", search.cv_results_["std_train_score"].mean(), search.cv_results_["std_train_score"].std())
print("mean_test_score: ", search.cv_results_["mean_test_score"].mean(), search.cv_results_["mean_test_score"].std())
print("std_test_score: ", search.cv_results_["std_test_score"].mean(),  search.cv_results_["std_test_score"].std())
print()

# MEASURE PERFORMANCE
print("\nResults best model fitted to all data")
print("Train accuracy score:", accuracy_score(y, best_model.predict(X)))

preprocess_features = get_transformer_feature_names(best_model.named_steps["preprocessor"])
plot_feature_importance_xgb(best_model.named_steps["model"], feature_names=preprocess_features)
plt.savefig(os.path.join(output_path, f'{PROJECT_NAME}_features_importance.png'), bbox_inches='tight')
PLOT_SHOW and plt.show()

# CREATE OUTPUT FILE
X_submission = submission_data[select_features]
y_pred_submission = best_model.predict(X_submission)
output = pd.DataFrame({target_key: submission_data.PassengerId, class_label: y_pred_submission})

output.to_csv(os.path.join(output_path, f'{PROJECT_NAME}_submission.csv'), index=False)
print("Your submission output was successfully saved!", len(output))
print("Program Duration: ", time.time() - start_time)
