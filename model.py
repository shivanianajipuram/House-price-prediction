import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("housing.csv")

print("Dataset Loaded Successfully")

# =========================
# HANDLE MISSING VALUES
# =========================

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# =========================
# ENCODE CATEGORICAL DATA
# =========================

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# =========================
# FEATURES AND TARGET
# =========================

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# =========================
# TRAIN TEST SPLIT
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(x_train, y_train)

# =========================
# MODEL EVALUATION
# =========================

predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# =========================
# SAVE MODEL
# =========================

pickle.dump(model, open("rf_model.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

print("Model Saved Successfully")