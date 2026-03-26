import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Title
st.title("🚢 Titanic Survival Predictor")
st.subheader("Predict survival based on passenger details")
st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg")

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Features
features = ['Pclass', 'Sex', 'Age']
target = 'Survived'
X = df[features].copy()
y = df[target]

# Encode Sex
X['Sex'] = LabelEncoder().fit_transform(X['Sex'])

# Preprocessing
num_features = ['Age']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features)
    ],
    remainder='passthrough'
)

# Model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'classifier__max_depth': [2, 3],
    'classifier__min_samples_split': [2],
    'classifier__criterion': ['gini']
}

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

# UI Inputs
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)

# Predict button
if st.sidebar.button("Predict"):

    sex_encoded = 1 if sex == "male" else 0

    input_data = pd.DataFrame([[pclass, sex_encoded, age]],
                              columns=['Pclass', 'Sex', 'Age'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Survived (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Did Not Survive (Probability: {probability:.2f})")