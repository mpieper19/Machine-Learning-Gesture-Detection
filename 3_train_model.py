from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

df = pd.read_csv('gesture_data.csv')
x = df.drop('ID', axis=1)
y = df['ID']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))


with open('gesture_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'gesture_model.pkl'")
