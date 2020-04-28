import pandas as pd
import seaborn as sns
import os

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

PROJECT_NAME = "titanic"

print(f"{PROJECT_NAME.upper()} KAGGLE COMPETITION")

print("Loading data ...")
print(os.getcwd())
main_path = os.path.join(os.getcwd(), "..")
print(main_path)
data_path = os.path.join(main_path, "data", PROJECT_NAME)
output_path = os.path.join(main_path, "output", PROJECT_NAME)
os.mkdir(output_path)
print(os.path.join(data_path, "gender_submission.csv"))

gender_submission = pd.read_csv(os.path.join(data_path, "gender_submission.csv"))
test = pd.read_csv(os.path.join(data_path, "test.csv"))
train = pd.read_csv(os.path.join(data_path, "train.csv"))

print(len(train))
train.head()

print(len(test))
test.head()

women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

features = ["Pclass", "Sex", "SibSp", "Parch"]

le_sex = LabelEncoder()
le_sex.fit(train["Sex"])
print(list(le_sex.classes_))
train["Sex_Encoded"] = le_sex.transform(train["Sex"])

le_emb = LabelEncoder()
le_emb.fit(train["Embarked"].astype("str"))
print(list(le_emb.classes_))
train["Embarked_Encoded"] = le_emb.transform(train["Embarked"].astype("str"))

sns.set(style="ticks", color_codes=True)
g = sns.pairplot(train, diag_kind="hist", hue="Survived")

#  Train Model for Submission
y_train = train["Survived"]

X_train = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Test accuracy score:", accuracy_score(y_train, y_pred_train))

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred_test})

# Save output file
output.to_csv(os.path.join(output_path, f'{PROJECT_NAME}_submission.csv'), index=False)
print("Your submission was successfully saved!", len(output))
