import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from copy import deepcopy
import featuretools as ft
import featuretools.variable_types as vtypes
import time

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from common.general import setup_environment
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
print("Dataset has {} entries and {} features".format(*train.shape))

# Define key labels
class_label = "Survived"
target_key = "PassengerId"
print("Submission cols:", sorted(submission_data.columns))
print("Train cols:", sorted(train.columns))
print(train.info())
print(train.head())

# DATA CLEANING
print("\n\nCleaning data ...")

# Clean up existing columns
train["Cabin_Sector"] = train["Cabin"].astype(str).str[0]
submission_data["Cabin_Sector"] = submission_data["Cabin"].astype(str).str[0]
print("Cabin_Sector (train) ", sorted(train["Cabin_Sector"].unique()))
print("Cabin_Sector (sub) ", submission_data["Cabin_Sector"].unique())
# todo fill in cabin sector based on class

train["Cabin_Locale"] = train["Cabin"].astype(str).str[1:]
submission_data["Cabin_Locale"] = submission_data["Cabin"].astype(str).str[1:]

train["Sex"] = train.Sex.apply(lambda x: 0.0 if x == "female" else 1.0)
submission_data["Sex"] = submission_data.Sex.apply(lambda x: 0.0 if x == "female" else 1.0)

# Identify and characterize categorical variables
categorical_cols = identify_categorical(train)
numerical_cols = list(train.columns.values)
numerical_cols = list(np.setdiff1d(numerical_cols, categorical_cols))
print("Numeric columns:", numerical_cols)

# Handle Missing Data (Impute)
print("\nHandling missing data...")
# Number of missing values in each column of training data
missing_val_count_by_column = (train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# Feature generation
print("\nGenerating features ...")

# Feature Selection
select_features_num = ["Pclass", "total_family", "Sex", ] # "Age", "Fare", "SibSp", "Parch"]
select_features_cat = ["Embarked", "Cabin_Sector",]
select_features = select_features_num + select_features_cat
print("Selected Features: ", select_features)

# Hand generate features
print("\nHand Generated Features")
f_name = "total_family"
numerical_cols.append(f_name)
train[f_name] = train["SibSp"] + train["Parch"]
submission_data[f_name] = submission_data["SibSp"] + submission_data["Parch"]

col_names = numerical_cols.copy()
col_names.remove("Survived")

for col in col_names:
    f_name = col + "_cbrt"
    print(f_name)
    numerical_cols.append(f_name)
    train[f_name] = np.cbrt(train[col])
    submission_data[f_name] = np.cbrt(submission_data[col])

print("Hand gen cols:", numerical_cols)

# Create FeatureTools entity set for automatic feature generation
# https://medium.com/dataexplorations/tool-review-can-featuretools-simplify-the-process-of-feature-engineering-5d165100b0c3
variable_types = { 'PassengerId': vtypes.Categorical,
      'Sex': vtypes.Categorical,
      'Pclass': vtypes.Categorical,
      'Embarked': vtypes.Categorical}
es = ft.EntitySet(id='Survivors')
es.entity_from_dataframe(entity_id='Passengers', dataframe=train[select_features + ["PassengerId"]], index='PassengerId',
                         variable_types=variable_types)
es = es.normalize_entity(base_entity_id='Passengers', new_entity_id='Pclass', index='Pclass')
print("\nFeatureTools input cols: ", es["Passengers"].variables)
feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='Passengers', max_depth=3, verbose=3, n_jobs=1)
# feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_names, include_unknown=False)
print("FeatureTools gen features:", feature_matrix.columns)

# perform same for submission
es_tst = ft.EntitySet(id='Survivors')
es_tst.entity_from_dataframe(entity_id='Passengers', dataframe=submission_data[select_features + ["PassengerId"]],
                             index='PassengerId')
es_tst = es_tst.normalize_entity(base_entity_id='Passengers', new_entity_id='Pclass', index='Pclass')
feature_matrix_tst = ft.calculate_feature_matrix(features=feature_names, entityset=es_tst)

#TODO feature combos?
#TODO Explore differences between train data and submission data
# TODO try semi supervised learning

# FEATURE REPORTING
# Print Data
print("\nTRAINING")
print(train.describe())

print("\nSUBMISSION")
print(submission_data.describe())

rate_women = (len(train.Sex) - sum(train.Sex))/len(train.Sex)
print("\n% of women who survived:", rate_women)
rate_men = sum(train.Sex)/len(train.Sex)
print("% of men who survived:", rate_men)
print(len(train[train[class_label] == 1]), len(train[train[class_label] == 0]))

# Plot data
print("\nPlotting data...")
sns.set(style="ticks", color_codes=True)
g_train = sns.pairplot(train, diag_kind="hist", hue="Survived")
plt.tight_layout()
SAVE_DATA and plt.savefig(os.path.join(output_path, f'{PROJECT_NAME}_allfeatures_pairplot.png'), bbox_inches='tight')
PLOT_SHOW and plt.show()
print()

# plot histograms
plot_feature_histogram(train, numerical_cols, class_col=class_label, hist_check_int=True)
SAVE_DATA and plt.savefig(os.path.join(output_path, f'{PROJECT_NAME}_features_histograms.png'), bbox_inches='tight')
PLOT_SHOW and plt.show()

# Define Model Training Inputs
y = train[class_label]
X = feature_matrix.copy()
X_submission = feature_matrix_tst.copy()

# Threshold for removing correlated variables
threshold = 0.7
# Absolute value correlation matrix
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Select columns with correlations above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]
print(f"Dropping collinear features > {threshold} ", collinear_features)
X = X.drop(columns=collinear_features)
X_submission = X_submission.drop(columns=collinear_features)

X_features_cat = identify_categorical(X)
X_features_num = list(X.columns)
X_features_num = [str(f) for f in np.setdiff1d(X_features_num, X_features_cat)]

print(f"\nInput {len(X.columns)} features: ", X.columns)
print("Numeric columns:", X_features_num)
print("Categorical columns:", X_features_cat)


# TRAIN MODEL PIPELINE
# Preprocessing Transformers
categorical_transformer = create_categorical_transformer(strategy="most_frequent")  #most_frequent
numerical_transformer = create_numerical_transformer(strategy="constant", fill_value=None)  #strategy constant, none
preprocessor = create_feature_preprocessor(numerical_transformer, X_features_num, categorical_transformer,
                                           X_features_cat)
scaler = StandardScaler()
pca = PCA()

# Train model
# Setting random state forces the classifier to produce the same result in each run
n_cv = 5  # cv=5 is default
scorer = "accuracy"
# model = LDA()
# model = DecisionTreeClassifier()
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

    # 'pca__n_components': [9, 10, 11, 12],

    # model parameters
    # XGBOOST
    'model__n_estimators': [25, 50, 75, 100, 125],  #75
    'model__max_depth': list(range(2, 7)),  #3
    'model__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09],  #0.01
    'model__colsample_bytree': [i/10. for i in range(5, 9, 2)],  #0.7
    'model__min_child_weight': list(range(5, 8)),   #7
    'model__subsample': [i/10. for i in range(5, 11, 2)],  #0.9
    # 'model__objective': list(range(2, 15)),
    # gamma, alpha, lambda  #finally, ensemble xgboost with multiple seeds may reduce variance
    'model__seed': [42],
}


# X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# temp_preprocessor = preprocessor.fit(X)
# X_temp = temp_preprocessor.transform(X)
# X_test = temp_preprocessor.transform(X_test)
# temp_scaler = scaler.fit(X)
# X_test = temp_scaler.transform(X_test)
# print("X_test:", X_test.shape, " X:", X.shape)
fit_params = {
                'model__eval_metric': "mae",
                # "model__num_boost_round": 999,  #  todo need to do this a separate training
                # "model__eval_set": [(X_test, y_test)],
                # "model__early_stopping_rounds": 10,  # todo need to do this as separate training
                "model__verbose": False
}

print("\nPerforming GridSearch on pipeline")
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=n_cv, scoring=scorer, return_train_score=True, refit=True,
                      verbose=1)
search.fit(X, y, **fit_params)
best_model = search.best_estimator_

print("\nSelected Features: ", X.columns)
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
print("Train confusion Matrix (tn, fp, fn, tp):", confusion_matrix(y, best_model.predict(X)).ravel())

# # Perform final fit
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# fit_params = {
#                 'model__eval_metric': "mae",
#                 # "model__num_boost_round": 999,  #  todo need to do this a separate training
#                 "model__eval_set": [(X_test, y_test)],
#                 "model__early_stopping_rounds": 10,  # todo need to do this as separate training
#                 "model__verbose": False
# }
# pipe_trf = Pipeline(pipe.steps[:-1])
# pipe_trf = pipe_trf.fit(pd.DataFrame(X_train))
# fit_params['model__eval_set'] = [(pipe_trf.transform(pd.DataFrame(X_train)),
#                                  pd.DataFrame(y_train)),
#                                 (pipe_trf.transform(pd.DataFrame(X_test)),
#                                  pd.DataFrame(y_test))]
# final_model = deepcopy(best_model)
# print(final_model)
# final_model.fit(X_train, y_train, **fit_params)

# # MEASURE PERFORMANCE
# print("\nResults best model fitted w/refit parameters")
# print("Train accuracy score:", accuracy_score(y_train, final_model.predict(X_train)))
# print("Train confusion Matrix (tn, fp, fn, tp):", confusion_matrix(y_train, final_model.predict(X_train)).ravel())
# print("Train accuracy score:", accuracy_score(y, final_model.predict(X_test)))
# print("Train confusion Matrix (tn, fp, fn, tp):", confusion_matrix(y_test, final_model.predict(X_test)).ravel())

#  PLOT PERFORMANCE
preprocess_features = get_transformer_feature_names(best_model.named_steps["preprocessor"])
print("\nFeature values:")
for i, f in enumerate(preprocess_features):
    print(f"f{i}={f}")

plot_feature_importance_xgb(best_model.named_steps["model"], feature_names=preprocess_features)
plt.savefig(os.path.join(output_path, f'{PROJECT_NAME}_features_importance.png'), bbox_inches='tight')
PLOT_SHOW and plt.show()

plt.figure(figsize=(20, 15))
xgb.plot_tree(best_model.named_steps["model"], ax=plt.gca())
plt.savefig(os.path.join(output_path, f'{PROJECT_NAME}_tree_plot.png'), bbox_inches='tight')
PLOT_SHOW and plt.show()

# CREATE OUTPUT FILE
y_pred_submission = best_model.predict(X_submission)
output = pd.DataFrame({target_key: submission_data.PassengerId, class_label: y_pred_submission})

output.to_csv(os.path.join(output_path, f'{PROJECT_NAME}_submission.csv'), index=False)
print("\nYour submission output was successfully saved!", len(output))
print("Program Duration: ", time.time() - start_time)
