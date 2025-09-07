# Predict Student Scores using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset: Study Hours vs Scores
data = {
    "Hours": [1.5, 2.5, 3.5, 4.5, 5, 6, 6.5, 7.5, 8, 9.25],
    "Scores": [20, 30, 50, 60, 62, 72, 75, 85, 88, 95]
}
df = pd.DataFrame(data)

# Split data
X = df[["Hours"]]
y = df["Scores"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Predicted Scores:", y_pred)
print("Actual Scores:", list(y_test))

# Plot
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study Hours vs Scores Prediction")
plt.show()
