import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)
print(len(train[train["Survived"] == 1]), len(train[train["Survived"] == 0]))

le_sex = LabelEncoder()
le_sex.fit(train["Sex"])
# print(list(le_sex.classes_))
train["Sex_Encoded"] = le_sex.transform(train["Sex"])
submission_data["Sex_Encoded"] = le_sex.transform(submission_data["Sex"])
print("sex", train["Sex"].head())

le_emb = LabelEncoder()
le_emb.fit(train["Embarked"].astype("str"))
# print(list(le_emb.classes_))
train["Embarked_Encoded"] = le_emb.transform(train["Embarked"].astype("str"))
submission_data["Embarked_Encoded"] = le_emb.transform(submission_data["Embarked"].astype("str"))
print(train.columns.values)
print(train.head())

# Plot data
sns.set(style="ticks", color_codes=True)
g_train = sns.pairplot(train, diag_kind="hist", hue="Survived")
plt.tight_layout()
PLOT_SHOW and plt.show()
print()

# Feature selection
features = ["Pclass", "Sex_Encoded", "SibSp", "Parch", 'Embarked_Encoded' ]#"Fare"] # "Age", ]#"Fare"]
print("Features: ", features)

# Split data into test train
test_ratio = 0.2
y = train["Survived"]
X = pd.get_dummies(train[features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

print("Train len:", len(X_train), "Test len:", len(X_test))

# Preprocessing
scaler = StandardScaler().fit(X_train)

# Train model
# TODO make this a pipeline
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(scaler.transform(X_train), y_train)
y_pred_train = model.predict(scaler.transform(X_train))
y_pred_test = model.predict(scaler.transform(X_test))

# Performance
#  you will find that your leaderboard score tends to be 2-5% lower because
#  the test.csv and train.csv have some major pattern differences
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
print("Train accuracy score:", accuracy_score(y_train, y_pred_train))
print("Test accuracy score:", accuracy_score(y_test, y_pred_test))
print("Test accuracy2 score:", (tp + tn)/(tp + tn + fp + fn))



# Predict and Save Submission File
X_submission = pd.get_dummies(submission_data[features])
y_pred_submission = model.predict(scaler.transform(X_submission))
output = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': y_pred_submission})

g_output = sns.pairplot(output, diag_kind="hist", hue="Survived")
plt.tight_layout()
PLOT_SHOW and plt.show()

output.to_csv(os.path.join(output_path, f'{PROJECT_NAME}_submission.csv'), index=False)
print("Your submission was successfully saved!", len(output))
