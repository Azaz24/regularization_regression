# 📊 Regularization in Regression

##  Project Overview
This project demonstrates the problem of **overfitting** in a regression model and how **regularization techniques (Ridge and Lasso)** help in improving model performance on unseen data.

---

##  Concepts Covered
- Overfitting
- Polynomial Features
- Linear Regression
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)

---

## 🛠️ Tech Stack
- Python
- NumPy
- Scikit-learn

---

## 📌 Problem Statement
In machine learning, models sometimes perform very well on training data but fail on new data. This is known as **overfitting**.  
The goal of this project is to:
- Demonstrate overfitting using a high-degree polynomial regression model
- Apply regularization techniques to reduce overfitting

---

## ⚙️ Steps Performed
1. Created a synthetic dataset
2. Split data into training and testing sets
3. Applied high-degree Polynomial Features to increase model complexity
4. Trained a Linear Regression model (to show overfitting)
5. Evaluated performance using R² score
6. Applied Ridge Regression
7. Applied Lasso Regression
8. Compared training and testing performance

---

## 📈 Results
- **Linear Regression**:
  - High training score
  - Low testing score → Overfitting

- **Ridge & Lasso Regression**:
  - Reduced training score
  - Improved testing score → Better generalization

---

## 📂 Project Structure


---

## 🎯 Conclusion
Regularization techniques like Ridge and Lasso help control overfitting by penalizing large coefficients, resulting in a more generalized and robust model.

---

## 👨‍💻 Author
Azaz Ahmed