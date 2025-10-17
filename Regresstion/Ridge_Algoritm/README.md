# 🚗 Car Price Prediction using Ridge Regression

## 📝 Project Description
This project implements a simple model to **predict car prices** using **Ridge Regression** in Python.  
The model uses the following features:

- **Year** of production (`Year`)
- **Mileage** (`KM`)
- **Fuel type** (`Fuel`) — 0: Petrol, 1: Diesel
- **Transmission type** (`Transmission`) — 0: Manual, 1: Automatic
- **Number of doors** (`Doors`)
- **Car brand** (`Brand`) — Encoded with **One-Hot Encoding**  

Categorical features like the car brand are converted using One-Hot Encoding so the model can handle them correctly.

---

## 🛠️ Technologies & Libraries
- Python 3.x
- Pandas
- NumPy
- scikit-learn

---

## ⚡ How to Use

1. **Install required libraries:**
```bash
pip install numpy pandas scikit-learn

Run the Python script:

```bash
python Car_Price.py

Check the predicted prices for new cars in the console output.

💻 Code Explanation

The dataset is created using a Pandas DataFrame.

The Brand column is transformed using One-Hot Encoding.

A Ridge Regression model with alpha=1 is trained on the numeric dataset.

You can add new car data in a DataFrame format and predict their prices easily.

📊 Sample Output
[26135.42841382 16183.28233088 28720.92929342]


This represents the predicted price for the new cars.

⚡ Notes

Ridge Regression helps the model handle noisy data better and prevents overfitting.

One-Hot Encoding ensures that categorical variables (like car brand) are treated independently without implying any order.