from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

# Load the pre-processed gesture data from CSV file
df = pd.read_csv('gesture_data.csv')

# Separate features (x) and labels (y)
x = df.drop('ID', axis=1)  # Features excluding the 'ID' column
y = df['ID']  # Target labels (gesture classes)

# Split data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42  # Ensures reproducibility
)

# Initialize the Gradient Boosting Classifier model
model = GradientBoostingClassifier(
    n_estimators=100,  # Number of boosting stages
    learning_rate=0.1,  # Step size for updating weights
    max_depth=3,  # Depth of each tree to control complexity
    random_state=42  # Reproducibility
)

# Train the model using the training dataset
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display the confusion matrix to analyze misclassifications
print(confusion_matrix(y_test, y_pred))

# Save the trained model to a file using pickle for later use
with open('gesture_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Confirm that the model has been saved
print("Model saved as 'gesture_model.pkl'")
