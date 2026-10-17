# Titanic Survival Prediction

## Overview
Titanic Survival Prediction is a Machine Learning project that predicts whether a passenger survived the Titanic disaster based on passenger information such as age, gender, passenger class, ticket fare, and family details.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle
- Jupyter Notebook

## Dataset
The dataset is from the Kaggle Titanic competition.

Dataset: https://www.kaggle.com/competitions/titanic/data

The dataset contains passenger information and a `Survived` target column:
- `0` — Did not survive
- `1` — Survived

## Project Workflow
1. Data loading and exploration
2. Exploratory Data Analysis (EDA)
3. Missing value handling
4. Feature selection and engineering
5. Categorical encoding using One-Hot Encoding
6. Train-test split
7. Data preprocessing using Scikit-learn pipelines
8. Model training using Logistic Regression
9. Model evaluation using accuracy, classification report, and confusion matrix
10. Model and preprocessor serialization using Pickle

## Features Used
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

## Model
Logistic Regression is used as the classification algorithm to predict passenger survival.

The trained model and preprocessing pipeline are saved together in `titanic_model.pkl` using Pickle for later use.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Run the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `titanic.ipynb` and run the cells to explore the data, preprocess features, train the model, and evaluate its performance.

## Project Structure

```text
Titanic-Survival-Prediction/
│
├── train.csv
├── titanic.ipynb
├── titanic_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Improvements
- Compare Logistic Regression with Random Forest and Decision Tree.
- Improve model performance through hyperparameter tuning.
- Build an interactive Streamlit web application.
- Add feature importance and model interpretation.

## Author
Rida Sohail

GitHub: https://github.com/ridasohail343-tech
