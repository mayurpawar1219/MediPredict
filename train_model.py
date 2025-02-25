import pandas as pd

# Load dataset
df = pd.read_csv("disease_symptom_data.csv")  # Change path if needed

# Display first few rows
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Convert symptoms text to numeric
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Symptoms"])  # Convert symptoms to feature vectors

# Encode disease labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Possible Disease"])

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Preprocessing Complete!")


from sklearn.ensemble import RandomForestClassifier

# Initialize model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

print("Model Training Complete!")


from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = rf_model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import joblib

# Save model
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model Saved Successfully!")

# Load model
rf_model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Test input
user_input = ["fever, cough, sore throat"]  # Example symptoms

# Convert input to vector
input_vector = vectorizer.transform(user_input)

# Predict disease
predicted_label = rf_model.predict(input_vector)
predicted_disease = label_encoder.inverse_transform(predicted_label)

print(f"Predicted Disease: {predicted_disease[0]}")
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Initialize GridSearch
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)

# Run GridSearch
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Use the best model
best_rf_model = grid_search.best_estimator_

# Evaluate best model
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy * 100:.2f}%")

from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Precision, Recall, F1-Score
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

# Print cross-validation accuracy
print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
