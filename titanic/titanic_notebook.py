import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Screen Setup
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Setup project variables and paths
PLOT_SHOW = False
PROJECT_NAME = "titanic"

print(f"{PROJECT_NAME.upper()} KAGGLE COMPETITION")

main_path = os.path.join(os.getcwd(), "..")
data_path = os.path.join(main_path, "data", PROJECT_NAME)
output_path = os.path.join(main_path, "output", PROJECT_NAME)

if not os.path.isdir(output_path):
    print("Making output folder")
    os.mkdir(output_path)

# Load up data
print("Loading data ...")
gender_submission = pd.read_csv(os.path.join(data_path, "gender_submission.csv"))
submission_data = pd.read_csv(os.path.join(data_path, "test.csv"))
train = pd.read_csv(os.path.join(data_path, "train.csv"))

# Data information
class_label = "Survived"
output_key = "PassengerId"
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

cols_encode = ["Sex", "Embarked", "Cabin_Sector"]
for encode in cols_encode:
    le = LabelEncoder()
    le.fit(train[encode].astype("str"))
    train[encode + "_Encoded"] = le.transform(train[encode].astype("str"))
    submission_data[encode + "_Encoded"] = le.transform(submission_data[encode].astype("str"))

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
features = ["Pclass", "Sex_Encoded", "SibSp", "Parch", "Embarked_Encoded",] # "Cabin_Sector_Encoded"]#"Fare"] # "Age", ]#"Fare"]
print("Selected Features: ", features)

# Split data into test train
test_ratio = 0.2
random_state = 42
y = train[class_label]
X = pd.get_dummies(train[features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)

print("Train len:", len(X_train), "Test len:", len(X_test))

# Preprocessing
scaler = StandardScaler().fit(X_train)

# Train model
# TODO make this a pipeline & tune parameters
# TODO perform random forest for important features
# TODO try semi supervised learning

# Setting random state forces the classifier to produce the same result in each run
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
model.fit(scaler.transform(X_train), y_train)
y_pred_train = model.predict(scaler.transform(X_train))
y_pred_test = model.predict(scaler.transform(X_test))

# Performance
#  you will find that your leaderboard score tends to be 2-5% lower because
#  the test.csv and train.csv have some major pattern differences
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
print("Train accuracy score:", accuracy_score(y_train, y_pred_train))
print("Test accuracy score:", accuracy_score(y_test, y_pred_test))
# print("Test accuracy2 score:", (tp + tn)/(tp + tn + fp + fn))

# Predict and Save Submission File
X_submission = pd.get_dummies(submission_data[features])
y_pred_submission = model.predict(scaler.transform(X_submission))
output = pd.DataFrame({output_key: submission_data.PassengerId, class_label: y_pred_submission})

output.to_csv(os.path.join(output_path, f'{PROJECT_NAME}_submission.csv'), index=False)
print("Your submission output was successfully saved!", len(output))
