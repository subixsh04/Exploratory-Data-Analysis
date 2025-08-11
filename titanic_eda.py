import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ========== 1) Load & inspect ==========
data = pd.read_csv('train.csv')

print(data.head())
data.info()

# Missing values overview
print("\nMissing counts per column:")
print(data.isnull().sum())

missing_percent = (data.isnull().sum() / len(data)) * 100
print("\nMissing % per column:")
print(missing_percent.round(2))

# ========== 2) Clean missing values ==========
# Drop Cabin (too many missing)
data = data.drop(columns=['Cabin'])

# Fill Age (numeric) with median
data['Age'] = data['Age'].fillna(data['Age'].median())

# Fill Embarked (categorical) with mode
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Verify missing are handled
print("\nMissing after cleaning:")
print(data.isnull().sum())

# ========== 3) Drop irrelevant / rename ==========
# Drop identifiers not helpful for prediction
data = data.drop(columns=['PassengerId', 'Ticket'])

# Rename Pclass for readability
data = data.rename(columns={'Pclass': 'PassengerClass'})

print("\nColumns after cleanup:")
print(data.columns.tolist())

# ========== 4) Basic EDA ==========
# Survival counts / percentage
print("\nSurvival counts:")
print(data['Survived'].value_counts())
print("\nSurvival %:")
print((data['Survived'].value_counts(normalize=True) * 100).round(2))

# Bar plot: survival count
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.show()

# Pie chart: survival distribution
survival_counts = data['Survived'].value_counts()
labels = ['Did Not Survive', 'Survived']
colors = ['lightcoral', 'skyblue']

plt.figure(figsize=(6, 6))
plt.pie(
    survival_counts,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    shadow=True,
    explode=[0, 0.1]
)
plt.title('Survival Distribution')
plt.axis('equal')
plt.show()

# Survival by gender
print("\nSurvival rate by gender:")
print(data.groupby('Sex')['Survived'].mean())

sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.show()

# Survival by passenger class
print("\nSurvival rate by passenger class:")
print(data.groupby('PassengerClass')['Survived'].mean())

sns.barplot(x='PassengerClass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.show()

# Survival by class + gender
print("\nSurvival rate by PassengerClass & Sex:")
print(data.groupby(['PassengerClass', 'Sex'])['Survived'].mean())

sns.barplot(x='PassengerClass', y='Survived', hue='Sex', data=data)
plt.title('Survival Rate by Passenger Class and Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.show()

# Age distribution
print("\nAge summary:")
print(data['Age'].describe())

sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

sns.boxplot(x='Survived', y='Age', data=data)
plt.title('Age vs Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()

# Fare distribution
sns.histplot(data['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.show()

sns.boxplot(x='Survived', y='Fare', data=data)
plt.title('Fare vs Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Fare')
plt.show()

# ========== 5) Preprocessing & split (Pipeline) ==========
features = ['PassengerClass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = data[features].copy()
y = data[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nShapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

print("\nTrain class balance (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("\nTest class balance (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))

num_cols = ['PassengerClass', 'Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Sex', 'Embarked']

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), cat_cols),
    ]
)

clf = Pipeline(steps=[
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

# ========== 6) Train & evaluate ==========
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Bonus: ROC–AUC and curve
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", round(auc, 3))
RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()

# ========== 7) Feature engineering ==========
# FamilySize: total people in a family group (self included)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# IsAlone: flag solo travelers
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# Title: extract honorific from Name (e.g., Mr, Miss, Mrs, Master, Dr, etc.)
# Pattern: everything between the comma and the period in the name string
data['Title'] = data['Name'].str.extract(r',\s*([^\.]+)\.')

# Group less common titles into 'Rare' and normalize a few variants
title_map = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
    'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
    'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
}
data['Title'] = data['Title'].replace(title_map)

# Drop Name now that Title is extracted (keeps feature set tidy)
data = data.drop(columns=['Name'])

# (Optional sanity checks)
print(data[['FamilySize','IsAlone','Title']].head())
print(data['Title'].value_counts())

# ========== 8) Update features & split ==========
features = ['PassengerClass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
    'FamilySize', 'IsAlone', 'Title']
target = 'Survived'

X = data[features].copy()
y = data[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nShapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

print("\nTrain class balance (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("\nTest class balance (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))

# Numerical vs categorical columns
num_cols = ['PassengerClass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
cat_cols = ['Sex', 'Embarked', 'Title']

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        # drop='first' can help with multicollinearity; keep None if full set is preffered
        ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), cat_cols),
    ]
)

clf = Pipeline(steps=[
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

# ========== 9) Train & evaluate with engineered features ==========
clf.fit(X_train, y_train)
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.3f}")
RocCurveDisplay.from_estimator(clf, X_test, y_test); plt.show()

# -----------------10) Comparing models --------------------------

# Swap model in pipeline
rf_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ))
])

# Train
rf_clf.fit(X_train, y_train)

# Predictions
y_pred_rf  = rf_clf.predict(X_test)
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

# Metrics
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Cross-validation score
cv_score = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')
print(f"\nCV Accuracy: {cv_score.mean():.3f} ± {cv_score.std():.3f}")

# --------------- 11) Tuning hyperparameters for Random forest
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 6, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Creating pipeline with RF
rf_pipeline = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", round(grid_search.best_score_, 3))

# Evaluating best model on test set
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

print("\nTest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

#Storing results
results = []

# Logistic Regression
log_reg_acc = accuracy_score(y_test, y_pred)
log_reg_auc = roc_auc_score(y_test, y_proba)
results.append(["Logistic Regression", log_reg_acc, log_reg_auc])

# Default Random Forest
rf_default = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestClassifier(random_state=42))
])
rf_default.fit(X_train, y_train)
y_pred_rf_def = rf_default.predict(X_test)
y_proba_rf_def = rf_default.predict_proba(X_test)[:, 1]
results.append(["Random Forest (Default)", 
                accuracy_score(y_test, y_pred_rf_def), 
                roc_auc_score(y_test, y_proba_rf_def)])

# Tuned Random Forest
results.append(["Random Forest (Tuned)", 
                accuracy_score(y_test, y_pred_rf), 
                roc_auc_score(y_test, y_proba_rf)])

# Convert to DataFrame
comparison_df = pd.DataFrame(results, columns=["Model", "Accuracy", "ROC-AUC"])
print(comparison_df)

