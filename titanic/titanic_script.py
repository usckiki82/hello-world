import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

from titanic.titanic_functions import setup_environment
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Setup project variables and paths
PLOT_SHOW = False
PROJECT_NAME = "titanic"

data_path, output_path = setup_environment(PROJECT_NAME)

# Load up data
print("Loading data ...")
gender_submission = pd.read_csv(os.path.join(data_path, "gender_submission.csv"))
submission_data = pd.read_csv(os.path.join(data_path, "test.csv"))
train = pd.read_csv(os.path.join(data_path, "train.csv"))

# Data information
class_label = "Survived"
target_key = "PassengerId"
print(sorted(train.columns))
print(sorted(submission_data.columns))

print("\nPrinting data info...")
women = train.loc[train.Sex == 'female'][class_label]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male'][class_label]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)
print(len(train[train[class_label] == 1]), len(train[train[class_label] == 0]))

# Data Cleaning
print("\nCleaning data ...")

train["Cabin_Sector"] = train["Cabin"].astype(str).str[0]
submission_data["Cabin_Sector"] = submission_data["Cabin"].astype(str).str[0]

train["Cabin_Local"] = train["Cabin"].astype(str).str[1:]
submission_data["Cabin_Local"] = submission_data["Cabin"].astype(str).str[1:]

# Get list of categorical variables
s = (train.dtypes == 'object')
object_cols = list(s[s].index)
# drop_X_train = X_train.select_dtypes(exclude=['object'])
# drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("Categorical variables:")
print(object_cols)

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

cols_encode = ["Sex", "Embarked", "Cabin_Sector"]

# numeric encoding
for encode in cols_encode:
    le = LabelEncoder()
    le.fit(train[encode].astype("str"))
    train[encode + "_Encoded"] = le.transform(train[encode].astype("str"))
    submission_data[encode + "_Encoded"] = le.transform(submission_data[encode].astype("str"))

# TODO try one hot encoing
# # Apply one-hot encoder to each column with categorical data
# OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
# OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
#
# # One-hot encoding removed index; put it back
# OH_cols_train.index = X_train.index
# OH_cols_valid.index = X_valid.index
#
# # Remove categorical columns (will replace with one-hot encoding)
# num_X_train = X_train.drop(object_cols, axis=1)
# num_X_valid = X_valid.drop(object_cols, axis=1)
#
# # Add one-hot encoded columns to numerical features
# OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
# OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Handle Missing Data
print("\nHandling missing data...")
# Deal with missing data fields
# Drop columns (not desired)
# Impute with Mean, 0 or mode
# Impute with Mean and add column indicating imputed rows
# Number of missing values in each column of training data
missing_val_count_by_column = (train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Imputation
# my_imputer = SimpleImputer()
# imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
# imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
# imputed_X_train.columns = X_train.columns
# imputed_X_valid.columns = X_valid.columns

# Feature generation
#TODO exponential combination
#TODO automated feature generation
#TODO Explore differences between train data and submission data

# Print Data
print("\nTRAINING")
print(train.describe())

print("\nSUBMISSION")
print(submission_data.describe())

# Plot data
sns.set(style="ticks", color_codes=True)
g_train = sns.pairplot(train, diag_kind="hist", hue="Survived")
plt.tight_layout()
PLOT_SHOW and plt.show()
print()

# TODO plot histograms

# Feature selection
features = ["Pclass", "Sex_Encoded", "SibSp", "Parch", "Embarked_Encoded", ]  # "Cabin_Sector_Encoded"]#"Fare"] # "Age", ]#"Fare"]
print("Selected Features: ", features)

# Split data into test train
test_ratio = 0.2
random_state = 42
y = train[class_label]
X = pd.get_dummies(train[features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)

print("Train len:", len(X_train), "Test len:", len(X_test))

# Preprocessing
scaler = StandardScaler()
pca = PCA()

# Train model
# TODO perform random forest for important features
# TODO try semi supervised learning
# TODO look at XGBOOST model

# Setting random state forces the classifier to produce the same result in each run
n_cv = 5  # cv=5 is default
scorer = "accuracy"
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
model = RandomForestClassifier(random_state=random_state)
pipe = Pipeline(steps=[
    ('scaler', scaler),
    # ('pca', pca),
    ('model', model)
])
param_grid = {
    # 'pca__n_components': [5, 15, 30, 45, 64],
    'model__n_estimators': [10, 20, 30, 40, 50, 60, 75, 100, 150, 200],
    'model__max_depth': list(range(2, 15)),
}

print("\nPerforming GridSearch on pipeline")
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=n_cv, scoring=scorer, return_train_score=True, refit=True)
search.fit(X, y)

print("Best parameter (%s score=%0.3f):" % (search.scorer_, search.best_score_))
print(search.best_params_)
best_model = search.best_estimator_
print(search.cv_results_.keys())
print("mean_train_score: ", search.cv_results_["mean_train_score"].mean(), search.cv_results_["mean_train_score"].std())
print("std_train_score: ", search.cv_results_["std_train_score"].mean(), search.cv_results_["std_train_score"].std())
print("mean_test_score: ", search.cv_results_["mean_test_score"].mean(), search.cv_results_["mean_test_score"].std())
print("std_test_score: ", search.cv_results_["std_test_score"].mean(),  search.cv_results_["std_test_score"].std())
print()

# Performance
print("\nResults best model fitted to all data")
# print(best_model)
print("Train accuracy score:", accuracy_score(y, best_model.predict(X)))

# Predict and Save Submission File
X_submission = pd.get_dummies(submission_data[features])
y_pred_submission = best_model.predict(X_submission)
output = pd.DataFrame({target_key: submission_data.PassengerId, class_label: y_pred_submission})

output.to_csv(os.path.join(output_path, f'{PROJECT_NAME}_submission.csv'), index=False)
print("Your submission output was successfully saved!", len(output))
