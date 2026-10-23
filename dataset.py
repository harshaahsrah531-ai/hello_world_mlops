# 1. Import necessary libraries
import pandas as pd
import joblib  # Used for saving and loading sklearn models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 2. Load the dataset and train the model
iris = load_iris()
X = iris.data[:, [2, 3]]  # Petal length (index 2) and petal width (index 3)
y = iris.target           # Target labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 3. Save the trained model to a file
model_location_tosave = 'D:\\varsha\\coding\\coding\\artifacts\\'
model_filename = 'iris_logistic_model.joblib'
joblib.dump(model, model_filename)
print(f"✅ Model successfully trained and saved to: '{model_location_tosave}/{model_filename}'\n")

# ==========================================
# SIMULATING A NEW SESSION / SCRIPT
# ==========================================

# 4. Reload the model from the file
loaded_model = joblib.load(model_filename)
print(f"📂 Model successfully reloaded from '{model_filename}'!\n")

# 5. Test multiple new custom flower samples using the reloaded model
# Format: [petal_length, petal_width]
new_flowers = [
    [1.4, 0.2],  # Sample 1 (Typically Setosa)
    [4.7, 1.4],  # Sample 2 (Typically Versicolor)
    [6.0, 2.0]   # Sample 3 (Typically Virginica)
]

print("--- Predictions Using the Reloaded Model ---")
for i, flower in enumerate(new_flowers, start=1):
    prediction = loaded_model.predict([flower])
    predicted_species = iris.target_names[prediction[0]]
    
    print(f"Flower #{i} -> Petal Length: {flower[0]} cm, Petal Width: {flower[1]} cm")
    print(f"Predicted Species: **{predicted_species.upper()}**\n")



# from sklearn.datasets import load_iris
# import pandas as pd

# # Load the Iris dataset
# iris = load_iris()
# # Access the features and target variable
# X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
# y = iris.target  # Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)

# # Print the feature names and target names

# print("Feature names:", pd.DataFrame(X))
# print("Target names:", pd.DataFrame(y))


# # print("Feature names:", iris.feature_names)
# # print("Target names:", iris.target_names)

# # # Print the first few samples in the dataset
# # print("First 5 samples:")
# # for i in range(5):
# #     print(f"Sample {i+1}: {X[i]} (Class: {y[i]}, Species: {iris.target_names[y[i]]})")