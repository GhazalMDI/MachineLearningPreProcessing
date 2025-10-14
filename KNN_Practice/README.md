# KNN_Practice
# KNN_Practice – Fruit Classification Example

A simple hands-on project to practice K-Nearest Neighbors (KNN) classification using fruit data. This project demonstrates the importance of feature scaling (standardization) in distance-based machine learning algorithms.

## Project Overview

The goal of this project is to predict the type of a new fruit (apple, plum, or watermelon) based on its weight and diameter.

**Key learning points:**

- How KNN uses Euclidean distance to classify new samples.
- The importance of standardizing features when their scales differ significantly.
- Visualizing decision boundaries for classification.

## Dataset

| Weight (grams) | Diameter (cm) | Fruit      |
|----------------|---------------|-----------|
| 150            | 7             | Apple     |
| 160            | 6             | Apple     |
| 150            | 8             | Apple     |
| 120            | 10            | Plum      |
| 130            | 9             | Plum      |
| 110            | 11            | Plum      |
| 950            | 20            | Watermelon|
| 800            | 18            | Watermelon|

## Dependencies

This project uses Python and the following libraries:

- numpy
- matplotlib
- scikit-learn

Install dependencies using:

```bash
pip install numpy matplotlib scikit-learn


