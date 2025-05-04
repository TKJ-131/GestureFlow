import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Load data
df = pd.read_csv('sign_data.csv')
X = df.iloc[:, 1:].values  # landmark positions
y = df.iloc[:, 0].values   # labels

# Split data to test performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save model
os.makedirs('model', exist_ok=True)
with open('model/sign_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to model/sign_classifier.pkl")
