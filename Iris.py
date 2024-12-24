import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset and train a simple model
iris = load_iris()
X, y = iris.data, iris.target
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Streamlit app
st.title("Iris Flower Classification")
st.write("This app predicts the type of Iris flower based on its features.")

# User input
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

if st.button("Predict"):
    # Prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = iris.target_names[prediction[0]]
    st.write(f"The predicted Iris species is: **{predicted_class}**")
