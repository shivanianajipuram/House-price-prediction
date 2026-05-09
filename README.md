# California House Price Prediction



A machine learning regression web application that predicts California house prices based on housing and demographic data. The model takes inputs such as location, rooms, population, income, and ocean proximity, and predicts median house value using trained regression algorithms.

The project demonstrates full ML workflow including data preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, and deployment using Streamlit.

---

# Dataset
File:
```
housing.csv
```

Features:
```
longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
population, households, median_income, ocean_proximity
```

Target:
```
median_house_value
```

---

# Data Preprocessing

- Missing values handled using column mean  
- Categorical encoding using LabelEncoder  
- Feature scaling using StandardScaler  
- Train-test split applied  
- K-Fold Cross Validation used  

Encoding:
```
<1H OCEAN -> 0
INLAND -> 1
ISLAND -> 2
NEAR BAY -> 3
NEAR OCEAN -> 4
```

---

# Algorithms Used

- Linear Regression  
- Lasso Regression  
- ElasticNet  
- K-Nearest Neighbors  
- Decision Tree Regressor  
- Support Vector Regressor  
- AdaBoost Regressor  
- Gradient Boosting Regressor  
- Random Forest Regressor  
- Extra Trees Regressor  

---

# Model

Random Forest Regressor


---

# Evaluation Metrics

- R² Score  
- Cross Validation Score  
- Prediction Accuracy  

---

# Tech Stack

Machine Learning: Python, Pandas, NumPy, Scikit-learn  
Visualization: Matplotlib, Seaborn  
Web App: Streamlit  
Model Saving: Pickle  
Version Control: Git, GitHub  

---

# Project Structure

```
California-House-Price-Prediction/
│
├── scaler.pkl
│── rf_model.pkl
├──housing.csv
├── app.py
├── model.py
├── requirements.txt
├── README.md
└── .gitignore
```


# How to Run

```bash
git clone https://github.com/shivanianajipuram/California-House-Price-Prediction.git

pip install -r requirements.txt
python model.py
streamlit run app.py
```
#live demo
```bash
https://house-price-prediction-pnzidehrmrczf6mb85ejz6.streamlit.app/
```
