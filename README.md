# salary-prediction---linear-Regression

---

# Case Study: Salary Prediction Using Simple Linear Regression

---

## 1. Introduction

Salary prediction is an important application of machine learning in human resource analytics. Organizations often want to estimate an employee’s salary based on measurable factors such as years of experience. Predicting salary accurately helps in fair compensation planning, budgeting, and recruitment decisions.

In this case study, a **Simple Linear Regression model** is developed to predict an employee’s salary based on **a single independent variable: years of experience**. Simple Linear Regression is used because it models the linear relationship between one independent variable and one dependent variable, making it easy to understand and interpret.

---

## 2. Case Study Objective

The objectives of this case study are:

* To predict employee salary based on years of experience
* To understand the linear relationship between experience and salary
* To evaluate the model using standard regression metrics
* To visualize the regression line and predictions

---

## 3. Dataset Description

The dataset contains two numerical variables:

| Feature         | Description                            |
| --------------- | -------------------------------------- |
| YearsExperience | Total years of professional experience |
| Salary          | Annual salary (target variable)        |

---

## 4. Methodology (Case Study Approach)

1. Import required libraries
2. Load and explore the dataset
3. Perform exploratory data analysis
4. Split the data into training and testing sets
5. Train a Simple Linear Regression model
6. Make predictions
7. Evaluate model performance
8. Visualize results

---

# 5. Code Commentary and Detailed Explanation

---

## Step 1: Importing Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
```

### Comment:

* **pandas** is used for data loading and manipulation.
* **numpy** supports numerical calculations.
* **matplotlib** is used for visualization.
* Warnings are suppressed for clean output.

---

## Step 2: Loading the Dataset

```python
data = pd.read_csv("Salary_Data.csv")
data.head()
```

### Comment:

* Loads the salary dataset from a CSV file.
* `head()` displays the first few records to understand data structure.

---

## Step 3: Understanding the Dataset

```python
data.info()
```

### Comment:

* Displays column names, data types, and null values.
* Confirms both variables are numerical.
* Ensures data suitability for Simple Linear Regression.

---

## Step 4: Checking for Missing Values

```python
data.isnull().sum()
```

### Comment:

* Checks for missing values.
* No missing values mean no preprocessing is required.

---

## Step 5: Descriptive Statistics

```python
data.describe()
```

### Comment:

* Provides summary statistics such as mean, min, max, and standard deviation.
* Helps understand data spread and variability.

---

## Step 6: Exploratory Data Analysis (EDA)

```python
plt.scatter(data["YearsExperience"], data["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()
```

### Comment:

* Scatter plot visualizes the relationship between experience and salary.
* Shows a positive linear trend, justifying the use of Linear Regression.

---

## Step 7: Defining Independent and Dependent Variables

```python
X = data[["YearsExperience"]]  # Independent variable
y = data["Salary"]             # Dependent variable
```

### Comment:

* `X` contains the single independent variable.
* `y` contains the target variable (salary).

---

## Step 8: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Comment:

* 80% data is used for training and 20% for testing.
* Helps evaluate model performance on unseen data.

---

## Step 9: Building the Simple Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### Comment:

* Initializes and trains the Simple Linear Regression model.
* Learns the best-fit line between experience and salary.

---

## Step 10: Model Parameters Interpretation

```python
print("Intercept:", model.intercept_)
print("Slope (Coefficient):", model.coef_[0])
```

### Comment:

* **Intercept**: Expected salary when experience is zero.
* **Slope**: Increase in salary for each additional year of experience.
* These parameters define the regression equation:

[
Salary = b_0 + b_1 \times Experience
]

---

## Step 11: Making Predictions

```python
y_pred = model.predict(X_test)
```

### Comment:

* Predicts salaries for test data.
* Used to compare actual vs predicted values.

---

## Step 12: Model Evaluation

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
```

### Comment:

* **R² Score** indicates how well experience explains salary variation.
* **MAE** shows average prediction error.
* **RMSE** penalizes larger errors more strongly.

---

## Step 13: Visualizing the Regression Line

```python
plt.scatter(X_train, y_train, label="Training Data")
plt.plot(X_train, model.predict(X_train), label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Simple Linear Regression")
plt.legend()
plt.show()
```

### Comment:

* Scatter plot shows actual data points.
* Regression line shows predicted relationship.
* Visual confirmation of model fit.

---

## 6. Results and Insights

* Salary increases linearly with years of experience.
* The model shows strong predictive performance.
* Experience is a significant factor in salary determination.
* Simple Linear Regression is effective for this problem.

---

## 7. Conclusion

This case study successfully demonstrates the use of **Simple Linear Regression** to predict employee salary based on years of experience. The model is easy to interpret, visually understandable, and performs well on evaluation metrics. It is suitable for HR analytics and educational purposes.

---

