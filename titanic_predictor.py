import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Select features and target
features = ['Pclass', 'Sex', 'Age']
target = 'Survived'
X = df[features].copy()
y = df[target]

# Encode 'Sex'
X['Sex'] = LabelEncoder().fit_transform(X['Sex'])

# Preprocessing pipeline
num_features = ['Age']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features)
    ],
    remainder='passthrough'  # Pclass and Sex go through as-is
)

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Grid search parameters 
param_grid = {
    'classifier__max_depth': [2, 3],  # Reduce depth to simplify visualization
    'classifier__min_samples_split': [2],
    'classifier__criterion': ['gini']
}


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

'''comment out the evaluation and visualization for results'''

# Evaluate on test set
'''y_pred = best_model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---- Tree Visualization ----
clf = best_model.named_steps['classifier']
plt.figure(figsize=(10, 6))
plot_tree(
    clf,
    feature_names=['Age', 'Pclass', 'Sex'],
    class_names=['Did Not Survive', 'Survived'],
    filled=True,
    rounded=True,
    max_depth=3  # Add this for extra safety
)
plt.title("Decision Tree")
plt.show()'''


# ---- Prediction function with graphical explanation ----
def predict_survival(pclass, sex, age):
    sex_encoded = 1 if sex.lower() == 'male' else 0
    input_data = pd.DataFrame([[pclass, sex_encoded, age]], columns=['Pclass', 'Sex', 'Age'])

    prediction = best_model.predict(input_data)[0]
    prediction_text = "Survived ✅" if prediction == 1 else "Did Not Survive ❌"

    print(f"\nPrediction: {prediction_text}")

# ---- Dynamic user input ----
try:
    print("\n🔍 Enter passenger details to predict survival:")
    pclass_input = int(input("Enter Pclass (1, 2, or 3): "))
    sex_input = input("Enter Sex (male/female): ")
    age_input = float(input("Enter Age: "))

    predict_survival(pclass=pclass_input, sex=sex_input, age=age_input)

except Exception as e:
    print(f"⚠️ Error: {e}")