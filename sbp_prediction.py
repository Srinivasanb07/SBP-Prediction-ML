import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load dataset
data = pd.read_csv("data.csv")

print("Dataset Preview:")
print(data.head())

# Step 2: Select input features (Age, BMI) and target (SBP)
X = data[["Age", "BMI"]]
y = data["SBP"]

# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Linear Regression model
model = LinearRegression()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict using test data
y_pred = model.predict(X_test)

# Step 7: Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Step 8: Predict for new input
age = 45
bmi = 28.5

new_data = pd.DataFrame([[age, bmi]], columns=["Age", "BMI"])
predicted_sbp = model.predict(new_data)

print("\nNew Prediction")
print("Age:", age)
print("BMI:", bmi)
print("Predicted SBP:", round(predicted_sbp[0], 2), "mmHg")